# clavenar-ai (Python) sequence diagrams

Five sequence diagrams covering the wire-level paths the Python SDK
can take: `clavenar_wrap` boot with the structural client + sync/async
fork, the async non-streaming inspection (Anthropic shown; OpenAI
Chat parallels), the async OpenAI Chat streaming choice-end gate
(Anthropic `content_block_stop` and the sync streams parallel),
`ClavenarPending.resolve` polling until decided, and the standalone
`inspect_realtime_function_call` helper for the OpenAI Realtime WS
surface. A request decision-tree flowchart closes the file.

The SDK is a faithful client of the same wire contract as the TS
sibling at [`clavenar-typescript-sdk`](https://github.com/clavenar/clavenar-typescript-sdk).
The diagrams emphasise the Python-specific shape (in-place
`.create` monkey-patch instead of a `Proxy` facade,
`inspect.iscoroutinefunction`-driven sync/async detection,
`httpx.AsyncClient` / `httpx.Client`, `asyncio.gather` vs a serial
sync loop, and the dual dict/attribute event access for
Pydantic-vs-raw streams).

## 1. `clavenar_wrap` — validate, detect, monkey-patch the right `.create`

`clavenar_wrap` runs once per client. It validates the option bag,
detects the client by `messages.create` (Anthropic) or
`chat.completions.create` (OpenAI), inspects that method with
`inspect.iscoroutinefunction` to pick the sync vs async path, then
monkey-patches `.create` in place. Python's attribute model doesn't
have a clean `Proxy` equivalent, so the in-place patch is the
cleanest seam — partners typically pass the wrapped client by
reference into LangChain / LlamaIndex / their own framework.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Partner code
    participant Wrap as clavenar_wrap
    participant V as _validate_options
    participant D as _detect_client
    participant Insp as inspect.iscoroutinefunction
    participant Patch as _wrap_kind_mode

    Caller->>Wrap: clavenar_wrap(client, ClavenarOptions(endpoint, token?, mode?, timeout_s?, on_verdict?, on_policy_error?, extra_headers?, retry?))
    Wrap->>V: _validate_options(opts)
    V->>V: opts.endpoint non-empty AND urlparse scheme+netloc both present
    V->>V: opts.timeout_s > 0
    V->>V: opts.mode in (enforce, observe)
    V->>V: retry.max_attempts >= 1 AND retry.base_delay_s >= 0
    alt any check fails
        V-->>Caller: raise ClavenarConfigError
    end
    Wrap->>D: _detect_client(client)
    alt client is None
        D-->>Caller: raise ClavenarConfigError
    end
    alt client.messages.create exists and callable
        D->>Insp: iscoroutinefunction(messages.create)
        Insp-->>D: True | False
        D-->>Wrap: ('anthropic', 'async' | 'sync')
    else client.chat.completions.create exists and callable
        D->>Insp: iscoroutinefunction(chat.completions.create)
        Insp-->>D: True | False
        D-->>Wrap: ('openai', 'async' | 'sync')
    else neither
        D-->>Caller: raise ClavenarConfigError — client must expose messages.create or chat.completions.create
    end
    Wrap->>Patch: dispatch to _wrap_{anthropic|openai}_{async|sync}
    Patch->>Patch: inner = client.<path>.create
    Patch->>Patch: define create_wrapped (closes over inner + opts)
    Patch->>Patch: client.<path>.create = create_wrapped  (in-place monkey-patch)
    Patch-->>Wrap: same client object
    Wrap-->>Caller: client (now wrapped)
    Note over Caller,Patch: every other attribute (client.beta, client.models, custom subclasses) untouched
```

## 2. Async non-streaming inspection — `asyncio.gather` parallel, submission-order raise

When the partner `await`s `client.messages.create(...)`, the patched
`create_wrapped` calls upstream, walks `content[]` for `tool_use`
blocks via `extract_tool_uses`, and runs `_inspect_all_async`. That
helper fans every call out concurrently via `asyncio.gather` over
`inspect_tool_use`, then consumes results in submission order so two
parallel denies always produce the same `ClavenarDenied.tool_name`
deterministically. The same pipeline applies to OpenAI Chat
(`extract_tool_calls` over `choices[].message.tool_calls`).

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Partner code
    participant Inner as create_wrapped (patched)
    participant Upstream as inner (real messages.create)
    participant Anth as api.anthropic.com
    participant Ext as extract_tool_uses
    participant Insp as _inspect_all_async
    participant T as inspect_tool_use (per call)
    participant Hx as httpx.AsyncClient
    participant L as clavenar-lite POST /mcp

    Caller->>Inner: await client.messages.create(model=..., tools=[...], messages=[...])
    Inner->>Upstream: inner(*args, **kwargs)
    Upstream->>Anth: POST /v1/messages
    Anth-->>Upstream: Message (content: [text, tool_use, tool_use, ...])
    Upstream-->>Inner: result
    Inner->>Inner: stream kwarg False AND result is not AsyncIterable — take non-streaming branch
    Inner->>Ext: extract_tool_uses(result)
    Ext-->>Inner: list[NormalizedToolCall] — id, name, input
    Inner->>Insp: _inspect_all_async(calls, opts)
    par asyncio.gather one(call) for call in calls
        Insp->>T: inspect_tool_use(call_1, opts)
        T->>T: retry loop up to opts.retry.max_attempts (default 3) with jittered exponential backoff
        T->>T: body = jsonrpc 2.0, method tools/call, params name + arguments, id
        T->>T: headers — Content-Type, Authorization Bearer (if token), opts.extra_headers (X-Clavenar-Demo-Prefix etc)
        T->>Hx: AsyncClient(timeout=opts.timeout_s).post(endpoint/mcp, json=body, headers, timeout)
        Hx->>L: POST /mcp
        L-->>Hx: 200 / 403 / 202 / 5xx
        Hx-->>T: response
        T->>T: read X-Clavenar-Correlation-Id header
        alt 200
            T-->>Insp: _Allow(correlation_id)
        else 403
            T->>T: _parse_deny_body — error security_violation + reasons + review_reasons + intent_category
            T-->>Insp: _Deny(reasons, review_reasons, intent_category, correlation_id)
        else 202
            T->>T: _parse_pending_body — corr = header OR body.correlation_id else ClavenarTransportError 202
            T-->>Insp: _Pending(correlation_id, review_reasons)
        else 5xx OR httpx.TimeoutException OR httpx.HTTPError
            T->>T: raise ClavenarTransportError — _is_retriable true for None status OR 5xx then sleep _backoff_s
            alt enforce mode (Sec 6 flowchart)
                T-->>Insp: raise after final attempt
            else observe mode
                Insp->>Insp: catch ClavenarTransportError as result
            end
        end
        Insp->>T: inspect_tool_use(call_2, opts)
        T-->>Insp: verdict_2 (parallel)
    end
    loop results in submission order calls[i]
        opt opts.on_verdict
            Insp->>Caller: await on_verdict(verdict, ClavenarVerdictContext(tool_name, tool_use_id, tool_input))
        end
        alt result is ClavenarTransportError (observe only)
            opt opts.on_policy_error
                Insp->>Caller: await on_policy_error(error, ctx)
            end
            Insp->>Insp: continue (treated as allow)
        else enforce AND verdict.kind == deny
            Insp-->>Caller: raise ClavenarDenied(tool_name, reasons, review_reasons, intent_category, correlation_id)
        else enforce AND verdict.kind == pending
            Insp-->>Caller: raise ClavenarPending(tool_name, correlation_id, review_reasons, poll_once closure)
        else allow OR observe
            Insp->>Insp: continue
        end
    end
    Inner-->>Caller: Message (only reached if no enforce-mode raise fired)
```

## 3. Async OpenAI Chat streaming — hold `finish_reason='tool_calls'` until inspection clears

`wrap_openai_chat_stream` is an async generator that yields upstream
chunks one-for-one. Tool deltas are accumulated per
`(choice_index, tool_index)` keyed against a `_ChoiceBufs` dict.
When a chunk's `choice.finish_reason == 'tool_calls'` arrives,
`_drain_openai_choice` materialises every accumulated call,
`_inspect_choice_batch` runs them concurrently via `asyncio.gather`,
verdicts are processed, and *only then* is the chunk yielded — so a
denied call raises before the partner ever sees the closing chunk.
The Anthropic stream wrap (`content_block_stop` per index) and both
sync stream wrappers follow the same pattern; only the
gather-vs-serial detail differs.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Partner code (async for)
    participant Gen as wrap_openai_chat_stream
    participant Up as upstream chunk iterator
    participant Evt as _evt (dict-or-attr access)
    participant Acc as _accumulate_openai (bufs _ChoiceBufs)
    participant Drain as _drain_openai_choice
    participant Batch as _inspect_choice_batch
    participant T as inspect_tool_use
    participant L as clavenar-lite POST /mcp

    Caller->>Gen: async for chunk in await client.chat.completions.create(stream=True, ...)
    loop every upstream chunk
        Gen->>Up: __anext__
        Up-->>Gen: ChatCompletionChunk (choices: [...])
        Gen->>Evt: choices = _evt(chunk, "choices") or [] (dict or Pydantic)
        loop every choice in chunk.choices
            Gen->>Evt: choice_idx, delta, deltas = _evt(choice, ...)
            alt deltas is list
                loop every d in deltas
                    Gen->>Acc: _accumulate_openai(bufs, choice_idx, d)
                    Acc->>Acc: setdefault _ChoiceBufs — setdefault _ToolBuf at tool_idx — buf.id, buf.name, buf.args_buf += partial
                end
            end
            alt _evt(choice, finish_reason) == 'tool_calls'
                Gen->>Gen: to_inspect.append(choice_idx)
            end
        end
        loop every choice_idx queued for inspection (BEFORE yielding chunk)
            Gen->>Drain: _drain_openai_choice(bufs, choice_idx)
            Drain->>Drain: pop _ChoiceBufs — for each _ToolBuf — if id or name missing raise ClavenarConfigError — json.loads(args_buf) or fall back to dict
            Drain-->>Gen: list[NormalizedToolCall]
            Gen->>Batch: _inspect_choice_batch(calls, opts, enforce)
            par asyncio.gather per call
                Batch->>T: inspect_tool_use(call, opts)
                T->>L: POST /mcp (same envelope and retry as Sec 2)
                L-->>T: 200 / 403 / 202 / 5xx
                T-->>Batch: ClavenarVerdict OR raise
            end
            loop results in submission order
                alt result is ClavenarTransportError (observe)
                    Batch->>Caller: await _fire_policy_error(e, ctx) — continue
                else enforce AND deny
                    Batch-->>Caller: raise ClavenarDenied — chunk never yielded
                else enforce AND pending
                    Batch-->>Caller: raise ClavenarPending — closure carries poll_once
                else
                    Batch->>Caller: await on_verdict if set — continue
                end
            end
        end
        Gen-->>Caller: yield chunk (only reached if no enforce raise fired)
    end
    Note over Gen,Caller: Anthropic content_block_stop and the two sync streams follow the same shape —<br/>sync variants use a serial for-loop instead of asyncio.gather
```

## 4. `ClavenarPending.resolve` — poll until decided, terminal vs transient errors

When enforce mode raises `ClavenarPending`, the partner catches it,
runs whatever side-work fits during the wait, then `await
pending.resolve(...)`. The loop polls
`GET /pending/{correlation_id}` every `poll_interval_s` (default 2s)
until the operator decides or the `time.monotonic()` deadline trips
at `timeout_s` (default 600s). Terminal transport failures (401,
404) re-raise immediately; everything else (5xx, network blips,
None view) is swallowed and the loop continues.

```mermaid
sequenceDiagram
    autonumber
    participant Partner as Partner try/except
    participant Pending as ClavenarPending.resolve
    participant Poll as self._poll_once (closure)
    participant T as poll_pending_once (or _sync)
    participant Hx as httpx.AsyncClient (or Client)
    participant L as clavenar-lite GET /pending/{id}

    Note over Partner: Sec 2 or Sec 3 raised ClavenarPending — partner caught it
    Partner->>Pending: await pending.resolve(poll_interval_s=2.0, timeout_s=600.0)
    Pending->>Pending: validate poll_interval_s > 0, timeout_s > 0
    Pending->>Pending: deadline = time.monotonic() + timeout_s
    loop while time.monotonic() < deadline
        Pending->>Poll: await self._poll_once()
        Poll->>T: poll_pending_once(correlation_id, opts) — async or sync flavour bound at raise time
        T->>T: headers — Authorization Bearer (if token), opts.extra_headers
        T->>Hx: get(endpoint/pending/{correlation_id}, headers, timeout=opts.timeout_s)
        Hx->>L: GET /pending/{correlation_id}
        alt 200
            L-->>Hx: ClavenarPendingView body
            Hx-->>T: response
            T->>T: _parse_pending_view — assert decision in (None, allow, deny)
            T-->>Poll: ClavenarPendingView
            Poll-->>Pending: view
            alt view.decision == allow
                Pending-->>Partner: return None
            else view.decision == deny
                Pending->>Pending: reasons = [decider_note] if non-empty else ['operator denied']
                Pending-->>Partner: raise ClavenarDenied(intent_category='PendingDenied', correlation_id, ...)
            else view.decision == None
                Pending->>Pending: not decided yet — fall through to sleep
            end
        else 401 or 404 (terminal)
            L-->>Hx: status
            Hx-->>T: response
            T-->>Poll: raise ClavenarTransportError(status)
            Poll-->>Pending: ClavenarTransportError with status 401 or 404
            Pending-->>Partner: re-raise immediately (terminal)
        else 5xx, httpx.TimeoutException, httpx.HTTPError
            L-->>Hx: error
            Hx-->>T: exception
            T-->>Poll: raise ClavenarTransportError
            Poll-->>Pending: ClavenarTransportError (other or no status)
            Pending->>Pending: swallow — view stays None
        end
        Pending->>Pending: remaining = deadline - time.monotonic() — break if <= 0
        Pending->>Pending: await asyncio.sleep(min(poll_interval_s, remaining))
    end
    Pending-->>Partner: raise ClavenarTransportError — clavenar pending {id} not decided within {timeout_s}s
```

## 5. OpenAI Realtime — one-shot `inspect_realtime_function_call`

The Realtime API is websocket-based; there is no `client.method()`
for `clavenar_wrap` to intercept. The partner drains the WS event
stream and runs each `response.function_call_arguments.done`
through `inspect_realtime_function_call`, which normalises the
event (parsing the JSON-encoded `arguments` string, falling back to
the raw string on parse failure so clavenar can still inspect the
malformed-args attempt) and runs one `inspect_tool_use`. The helper
returns the verdict — caller decides how to react (return a
synthesised `function_call_output`, hold the pump, surface to
operator).

```mermaid
sequenceDiagram
    autonumber
    participant Partner as WS message pump
    participant Guard as is_realtime_function_call_done
    participant Helper as inspect_realtime_function_call
    participant Norm as normalize_realtime_function_call
    participant T as inspect_tool_use
    participant L as clavenar-lite POST /mcp
    participant Rt as OpenAI Realtime WS

    Note over Rt,Partner: each tool call arrives as:<br/>response.output_item.added — carries call_id + name<br/>then 1..N response.function_call_arguments.delta<br/>then exactly one response.function_call_arguments.done with the full arguments
    Rt-->>Partner: evt = { type: response.function_call_arguments.done, call_id, arguments, name }
    Partner->>Guard: is_realtime_function_call_done(evt)
    Guard-->>Partner: True (type matches AND call_id/arguments/name are str)
    Partner->>Helper: await inspect_realtime_function_call(evt, opts)
    Helper->>Norm: normalize_realtime_function_call(evt)
    Norm->>Norm: input = json.loads(evt.arguments) — on JSONDecodeError input = evt.arguments (raw string)
    Norm-->>Helper: NormalizedToolCall(id=evt.call_id, name=evt.name, input)
    Helper->>T: inspect_tool_use(call, opts)
    T->>L: POST /mcp (same envelope and retry as Sec 2)
    L-->>T: 200 / 403 / 202 / 5xx
    T-->>Helper: ClavenarVerdict (no raise on deny — caller decides)
    Helper-->>Partner: verdict
    alt verdict.kind == 'deny'
        Partner->>Rt: send { type: conversation.item.create, item: { type: function_call_output, call_id, output: denied... } }
        Partner->>Partner: continue WS pump (skip the tool dispatch)
    else verdict.kind == 'pending'
        Partner->>Partner: surface to operator OR hold WS pump OR synthesise placeholder output — caller's choice
    else allow
        Partner->>Partner: dispatch the tool handler normally
    end
```

## 6. Request decision tree (flowchart)

A single `create()` invocation through the wrapped client fans out
across five orthogonal knobs: sync vs async client, response shape
(non-streaming vs streaming), per-call verdict, enforcement mode,
and transport health. The flowchart captures the final outcomes —
pass-through, `ClavenarDenied`, `ClavenarPending`, `ClavenarTransportError`,
or a `ClavenarConfigError` raised before any of those.

```mermaid
flowchart TD
    Start([wrapped.create call]) --> Cfg{validate_options AND<br/>_detect_client OK?}
    Cfg -->|no| Cerr[raise ClavenarConfigError<br/>partner bug — bad endpoint, missing client, sync callback on async wrap, etc.]
    Cfg -->|yes| Fork{sync vs async path?}
    Fork -->|async| Up1[await inner upstream call]
    Fork -->|sync| Up2[inner upstream call]

    Up1 --> Shape1{stream=True OR result is AsyncIterable?}
    Up2 --> Shape2{stream=True OR _is_iterable_non_message?}

    Shape1 -->|no| Sync1[extract NormalizedToolCalls<br/>walk content tool_use OR choices message tool_calls]
    Shape1 -->|yes| Stream1[wrap async generator<br/>accumulate per index<br/>inspect on content_block_stop OR finish_reason tool_calls]

    Shape2 -->|no| Sync2[extract NormalizedToolCalls — same shape]
    Shape2 -->|yes| Stream2[wrap sync generator — same shape — serial inspect instead of gather]

    Sync1 --> Loop1[_inspect_all_async — asyncio.gather per call<br/>consume in submission order]
    Stream1 --> Loop1
    Sync2 --> Loop2[_inspect_all_sync — serial for-loop<br/>callbacks must be sync OR raise ClavenarConfigError]
    Stream2 --> Loop2

    Loop1 --> Insp{inspect_tool_use outcome}
    Loop2 --> Insp

    Insp -->|transport timeout / 5xx after retries / httpx.HTTPError| Tport{mode}
    Tport -->|enforce| Te[raise ClavenarTransportError — fail-closed]
    Tport -->|observe| Toe[await on_policy_error if set<br/>continue — treated as allow]

    Insp -->|verdict| Mode{mode}
    Mode -->|observe| Obs[await on_verdict if set<br/>continue]
    Mode -->|enforce| Vk{verdict.kind}
    Vk -->|allow| Ok[await on_verdict if set<br/>continue]
    Vk -->|deny| Dn[raise ClavenarDenied — tool_name, reasons, review_reasons, intent_category, correlation_id]
    Vk -->|pending| Pn[raise ClavenarPending — correlation_id, review_reasons, async closure poll_once]

    Pn --> Resolve[partner catches → await pending.resolve poll loop]
    Resolve --> Rd{decision}
    Rd -->|allow within deadline| Rok[return None — partner re-runs original create]
    Rd -->|deny within deadline| Rdn[raise ClavenarDenied — intent_category PendingDenied + decider_note]
    Rto[raise ClavenarTransportError — not decided within timeout_s]
    Rd -->|deadline trip| Rto
    Rd -->|401 or 404 terminal| Rterm[raise ClavenarTransportError immediately]

    Obs --> Pass[response or stream chunks pass through to partner]
    Ok --> Pass
    Toe --> Pass
    Te --> End([raises land at partner await/call])
    Dn --> End
    Pn --> End
    Cerr --> End
    Pass --> End
    Rok --> End
    Rdn --> End
    Rto --> End
    Rterm --> End
```
