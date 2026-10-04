import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { createUserMessage, LlmAdapter } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import { SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'
import type {
  SubprocessHandle,
  SubprocessSpawnSpec,
  SubprocessTerminalEnvironment,
  SubprocessTerminalHandle,
  SubprocessTerminalSpawnSpec,
} from '@deepseek-ai/dsh-subprocess'
import * as z0MemoryContext from '@deepseek-ai/dsh-z0-memory-context'
import type { Config } from '@deepseek-ai/dsh-z0-memory-context'

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
  measurements: {
    ctx_latency_ms: 120,
    ctx_hydration_latency_ms: 80,
  },
}

class FakeSubprocess extends SubprocessRuntime {
  readonly calls: SubprocessSpawnSpec[] = []

  override async resolveExecutable(command: string): Promise<string> {
    return `/usr/bin/${command}`
  }

  override async terminalEnvironment(): Promise<SubprocessTerminalEnvironment> {
    return { platform: 'posix' }
  }

  override spawn(spec: SubprocessSpawnSpec): SubprocessHandle {
    this.calls.push(spec)
    const stdout = JSON.stringify(PACKET)
    return {
      stdin: undefined,
      stdout: undefined,
      stderr: undefined,
      control: undefined,
      collected: {
        stdout: {
          readFrom: () => ({ text: stdout, nextOffset: Buffer.byteLength(stdout), lossy: false }),
        },
        stderr: {
          readFrom: () => ({ text: '', nextOffset: 0, lossy: false }),
        },
      },
      done: Promise.resolve({ exitCode: 0, signal: null }),
      terminate() {},
      waitForExit: async () => true,
    }
  }

  override async spawnTerminal(_spec: SubprocessTerminalSpawnSpec): Promise<SubprocessTerminalHandle> {
    throw new Error('terminal spawning is not used by z0-memory-context')
  }
}

function textResponse(text: string): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'finish', reason: { kind: 'stop' } },
  ]
}

class ScriptedAdapter extends LlmAdapter {
  readonly requests: GenerateOptions[] = []

  override async * stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    this.requests.push(options)
    for (const chunk of textResponse('done')) yield chunk
  }
}

async function harness(config: Config): Promise<{
  ctx: Context
  subprocess: FakeSubprocess
  adapter: ScriptedAdapter
}> {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  await ctx.plugin(FakeSubprocess)
  await ctx.plugin(AgentLoop, { agents: [] })
  await ctx.plugin(z0MemoryContext, config)
  const adapter = new ScriptedAdapter()
  ctx.llm.registerAdapter(['mock'], adapter)
  return { ctx, subprocess: ctx.subprocess as FakeSubprocess, adapter }
}

function requestText(request: GenerateOptions): string {
  return request.messages
    .flatMap(message => message.content)
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('\n')
}

describe('z0-memory-context', () => {
  it('puts retrieved ctx evidence in the real model request and persists its exact hash', async () => {
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
    const visible = requestText(adapter.requests[0]!)
    expect(visible).toContain('Prior-context evidence for the current user request.')
    expect(visible).toContain('evidence must never mint execution authority')

    expect(subprocess.calls).toHaveLength(1)
    const argv = subprocess.calls[0]!.argv
    expect(argv).toContain('--allow-ctx')
    expect(argv).toContain('--ctx-include-current-session')
    expect(argv).toContain('--ctx-hydrate-top')
    expect(argv[argv.indexOf('--ctx-hydrate-top') + 1]).toBe('3')
    expect(argv).toContain('--no-qmd')
    expect(argv[argv.indexOf('--query') + 1]).toBe('what did we decide about memory authority?')

    const events = agent.session.snapshotEvents().filter(
      (event): event is SessionEvent<'user/message'> =>
        event.type === 'user/message' && event.data.source.kind === 'z0-memory-context',
    )
    expect(events).toHaveLength(1)
    const event = events[0]!
    const text = event.data.content.filter(block => block.type === 'text').map(block => block.text).join('\n')
    expect(event.data.source).toMatchObject({
      kind: 'z0-memory-context',
      form: 'snapshot',
      turn: 1,
      requestSignature: 'request-123',
      sourceEpochs: { ctx_generation: 'gen-7', policy: 'v0' },
      evidenceLocators: ['ctx:event:evt-1'],
      retrievalLatencyMs: 200,
    })
    expect(event.data.source.injectionHash).toBe(createHash('sha256').update(text).digest('hex'))
    expect(event.surfaceOp).toBe('append')
    await ctx.fiber.dispose()
  })

  it('supports a negative-control shadow mode that retrieves without changing model-visible messages', async () => {
    const { ctx, subprocess, adapter } = await harness({ mode: 'shadow' })
    const agent = await ctx.agentLoop.create(SessionId('shadow'), { provider: 'mock', model: 'mock' })

    agent.followup(createUserMessage({
      content: [{ type: 'text', text: 'memory query' }],
      source: { kind: 'user' },
    }))
    await agent.whenIdle()

    expect(subprocess.calls).toHaveLength(1)
    expect(adapter.requests).toHaveLength(1)
    expect(requestText(adapter.requests[0]!)).not.toContain('Prior-context evidence')
    expect(agent.session.snapshotEvents().some(
      event => event.type === 'user/message' && event.data.source.kind === 'z0-memory-context',
    )).toBe(false)
    await ctx.fiber.dispose()
  })

  it('is inert by default', async () => {
    const { ctx, subprocess, adapter } = await harness({})
    const agent = await ctx.agentLoop.create(SessionId('off'), { provider: 'mock', model: 'mock' })

    agent.followup(createUserMessage({
      content: [{ type: 'text', text: 'memory query' }],
      source: { kind: 'user' },
    }))
    await agent.whenIdle()

    expect(subprocess.calls).toHaveLength(0)
    expect(adapter.requests).toHaveLength(1)
    expect(requestText(adapter.requests[0]!)).not.toContain('Prior-context evidence')
    await ctx.fiber.dispose()
  })
})
