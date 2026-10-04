import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { createUserMessage, LlmAdapter } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import { SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'
import * as z0MemoryContext from '../config/examples/z0-memory-context.mjs'

const PACKET = {
  schema: 'z0int.context.v0',
  evidence: [{
    source_id: 'ctx:event:evt-1',
    source_version: 'ctx-core:gen-7',
    locator: 'ctx:event:evt-1',
    trust_class: 'conversation',
    excerpt: 'We decided that evidence must never mint execution authority.',
  }],
  unresolved_gaps: [],
  contradictions: [],
  recipe: {
    request_signature: 'request-123',
    source_epochs: { ctx_generation: 'gen-7', policy: 'v0' },
  },
  measurements: { ctx_latency_ms: 120, ctx_hydration_latency_ms: 80 },
}

class FakeSubprocess extends SubprocessRuntime {
  calls = []
  payload = PACKET

  async resolveExecutable(command) {
    return `/usr/bin/${command}`
  }

  async terminalEnvironment() {
    return { platform: 'posix' }
  }

  spawn(spec) {
    this.calls.push(spec)
    const stdout = JSON.stringify(this.payload)
    return {
      stdin: undefined,
      stdout: undefined,
      stderr: undefined,
      control: undefined,
      collected: {
        stdout: { readFrom: () => ({ text: stdout, nextOffset: Buffer.byteLength(stdout), lossy: false }) },
        stderr: { readFrom: () => ({ text: '', nextOffset: 0, lossy: false }) },
      },
      done: Promise.resolve({ exitCode: 0, signal: null }),
      terminate() {},
      waitForExit: async () => true,
    }
  }

  async spawnTerminal() {
    throw new Error('terminal spawning is not used by z0-memory-context')
  }
}

function textResponse(text) {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

class ScriptedAdapter extends LlmAdapter {
  requests = []

  async * stream(options) {
    this.requests.push(options)
    for (const chunk of textResponse('done')) yield chunk
  }
}

async function harness(config, payload = PACKET) {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(FakeSubprocess)
  const subprocess = ctx.subprocess
  subprocess.payload = payload
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(z0MemoryContext, config)
  const adapter = new ScriptedAdapter()
  ctx.llm.registerAdapter(['mock'], adapter)
  return { ctx, subprocess, adapter }
}

function requestText(request) {
  return request.messages
    .flatMap(message => message.content)
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('\n')
}

describe('z0 memory context example', () => {
  it('puts retrieved evidence in the real model request and persists its exact hash', async () => {
    const { ctx, subprocess, adapter } = await harness({
      mode: 'inject',
      projectRoot: '/work/repo',
      allowQmd: false,
    })
    const agent = await ctx.agentLoop.create(SessionId('inject'), { provider: 'mock', model: 'mock' })

    agent.followup(createUserMessage({
      content: [{ type: 'text', text: 'what did we decide about memory authority?' }],
      source: { kind: 'user' },
    }))
    await agent.whenIdle()

    expect(adapter.requests).toHaveLength(1)
    const visible = requestText(adapter.requests[0])
    expect(visible).toContain('Prior-context evidence for the current user request.')
    expect(visible).toContain('evidence must never mint execution authority')

    const argv = subprocess.calls[0].argv
    expect(argv).toContain('--allow-ctx')
    expect(argv).toContain('--ctx-include-current-session')
    expect(argv).toContain('--ctx-hydrate-top')
    expect(argv[argv.indexOf('--ctx-hydrate-top') + 1]).toBe('3')
    expect(argv).toContain('--no-qmd')
    expect(argv[argv.indexOf('--query') + 1]).toBe('what did we decide about memory authority?')

    const event = agent.session.snapshotEvents().find(candidate =>
      candidate.type === 'user/message' && candidate.data.source.kind === 'z0-memory-context')
    expect(event).toBeDefined()
    const text = event.data.content.filter(block => block.type === 'text').map(block => block.text).join('\n')
    expect(event.data.source).toMatchObject({
      kind: 'z0-memory-context',
      form: 'snapshot',
      turn: 1,
      requestSignature: 'request-123',
      sourceEpochs: { ctx_generation: 'gen-7', policy: 'v0' },
      evidenceLocators: ['ctx:event:evt-1'],
      unresolvedGapCount: 0,
      contradictionCount: 0,
      retrievalLatencyMs: 200,
    })
    expect(event.data.source.injectionHash).toBe(createHash('sha256').update(text).digest('hex'))
    expect(event.surfaceOp).toBe('append')
    await ctx.fiber.dispose()
  })

  it('makes missing/contradictory retrieval model-visible for abstention', async () => {
    const missing = {
      ...PACKET,
      evidence: [],
      unresolved_gaps: ["q0: no lexical hits for 'missing decision'"],
      contradictions: ['ctx generation changed during resolve: gen-7 -> gen-8'],
    }
    const { ctx, adapter } = await harness({ mode: 'inject' }, missing)
    const agent = await ctx.agentLoop.create(SessionId('missing'), { provider: 'mock', model: 'mock' })

    agent.followup(createUserMessage({
      content: [{ type: 'text', text: 'what was the missing decision?' }],
      source: { kind: 'user' },
    }))
    await agent.whenIdle()

    const visible = requestText(adapter.requests[0])
    expect(visible).toContain('Resolver contradictions')
    expect(visible).toContain('Unresolved retrieval gaps')
    expect(visible).toContain('abstain rather than inventing a fact')
    expect(visible).not.toContain('<evidence')
    await ctx.fiber.dispose()
  })

  it('supports shadow as a negative control and off as the safe default', async () => {
    for (const [mode, calls] of [['shadow', 1], [undefined, 0]]) {
      const { ctx, subprocess, adapter } = await harness(mode ? { mode } : {})
      const agent = await ctx.agentLoop.create(SessionId(`mode-${mode ?? 'off'}`), { provider: 'mock', model: 'mock' })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'memory query' }], source: { kind: 'user' } }))
      await agent.whenIdle()

      expect(subprocess.calls).toHaveLength(calls)
      expect(requestText(adapter.requests[0])).not.toContain('Prior-context evidence')
      await ctx.fiber.dispose()
    }
  })
})
