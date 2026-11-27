/**
 * A local OTLP/HTTP endpoint on a raw TCP server, so tests control the socket exactly:
 * hanging responses, dropped connections after the body was read (lost acknowledgements),
 * trickling bodies, redirects, `Retry-After`, and observing client disconnects.
 */
import { createServer, type Socket } from "node:net";

export type ReceivedRequest = {
  readonly index: number;
  readonly path: string;
  /** Lower-cased header names; repeated headers joined with ", ". */
  readonly headers: Readonly<Record<string, string>>;
  readonly body: Buffer;
  readonly receivedAt: number;
  /** Set when the client closed the connection before a response was completed. */
  disconnected: boolean;
};

export type Reply =
  | {
      readonly status: number;
      readonly headers?: Readonly<Record<string, string>>;
      readonly body?: Uint8Array | string;
      readonly delayMs?: number;
    }
  /** Never answers; the connection stays open until the client gives up. */
  | { readonly hang: true }
  /** Reads the request, then closes the connection without a response. */
  | { readonly destroy: true }
  /** Sends the status and `first`, then the rest of the body slowly. */
  | {
      readonly trickle: {
        readonly status: number;
        readonly first: Uint8Array;
        readonly rest: Uint8Array;
        readonly delayMs: number;
      };
    };

export type Handler = (request: ReceivedRequest) => Reply | Promise<Reply>;

export type TestServer = {
  readonly url: string;
  readonly requests: ReceivedRequest[];
  /** Requests whose path is `path`. */
  received(path: string): ReceivedRequest[];
  /** Resolves when `predicate` holds (polling), or rejects after `timeoutMs`. */
  waitFor(predicate: () => boolean, timeoutMs?: number): Promise<void>;
  close(): Promise<void>;
};

const REASONS: Readonly<Record<number, string>> = {
  200: "OK",
  202: "Accepted",
  301: "Moved Permanently",
  302: "Found",
  307: "Temporary Redirect",
  308: "Permanent Redirect",
  400: "Bad Request",
  429: "Too Many Requests",
  500: "Internal Server Error",
  502: "Bad Gateway",
  503: "Service Unavailable",
  504: "Gateway Timeout",
};

const head = (
  status: number,
  headers: Readonly<Record<string, string>>,
  length?: number,
): string => {
  const lines = [`HTTP/1.1 ${status} ${REASONS[status] ?? "Status"}`, "Connection: close"];
  if (length !== undefined) lines.push(`Content-Length: ${length}`);
  for (const [name, value] of Object.entries(headers)) lines.push(`${name}: ${value}`);
  return `${lines.join("\r\n")}\r\n\r\n`;
};

export const startServer = async (
  handler: Handler = () => ({ status: 200 }),
  /** Listen on this port (for example one a test released earlier); default: any free port. */
  port = 0,
): Promise<TestServer> => {
  const requests: ReceivedRequest[] = [];
  const sockets = new Set<Socket>();
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const later = (ms: number, run: () => void): void => {
    const timer = setTimeout(() => {
      timers.delete(timer);
      run();
    }, ms);
    timers.add(timer);
  };

  const server = createServer((socket) => {
    sockets.add(socket);
    let buffer = Buffer.alloc(0);
    let handled = false;
    let responded = false;
    let current: ReceivedRequest | undefined;
    socket.on("error", () => undefined);
    socket.on("close", () => {
      sockets.delete(socket);
      if (current !== undefined && !responded) current.disconnected = true;
    });
    const finish = (data: string | Uint8Array): void => {
      if (socket.destroyed) return;
      responded = true;
      socket.end(data);
    };
    socket.on("data", (chunk: Buffer) => {
      if (handled) return;
      buffer = Buffer.concat([buffer, chunk]);
      const end = buffer.indexOf("\r\n\r\n");
      if (end < 0) return;
      const lines = buffer.subarray(0, end).toString("latin1").split("\r\n");
      const headers: Record<string, string> = {};
      for (const line of lines.slice(1)) {
        const colon = line.indexOf(":");
        if (colon <= 0) continue;
        const name = line.slice(0, colon).trim().toLowerCase();
        const value = line.slice(colon + 1).trim();
        headers[name] = headers[name] === undefined ? value : `${headers[name]}, ${value}`;
      }
      const length = Number(headers["content-length"] ?? "0");
      if (buffer.length < end + 4 + length) return;
      handled = true;
      const [, path = ""] = (lines[0] ?? "").split(" ");
      const request: ReceivedRequest = {
        index: requests.length,
        path,
        headers,
        body: Buffer.from(buffer.subarray(end + 4, end + 4 + length)),
        receivedAt: performance.now(),
        disconnected: false,
      };
      current = request;
      requests.push(request);
      void Promise.resolve(handler(request)).then((reply) => {
        if ("hang" in reply) return;
        if ("destroy" in reply) {
          socket.destroy();
          return;
        }
        if ("trickle" in reply) {
          const { status, first, rest, delayMs } = reply.trickle;
          socket.write(head(status, {}, first.length + rest.length));
          socket.write(first);
          later(delayMs, () => finish(rest));
          return;
        }
        const body =
          typeof reply.body === "string" ? Buffer.from(reply.body) : Buffer.from(reply.body ?? []);
        const send = (): void => {
          if (socket.destroyed) return;
          socket.write(head(reply.status, reply.headers ?? {}, body.length));
          finish(body);
        };
        if (reply.delayMs === undefined) send();
        else later(reply.delayMs, send);
      });
    });
  });
  // A released port can stay busy for a moment; the bound only detects one that never frees up.
  const until = performance.now() + 10_000;
  for (;;) {
    const error = await new Promise<NodeJS.ErrnoException | undefined>((resolve) => {
      const failed = (e: NodeJS.ErrnoException): void => resolve(e);
      server.once("error", failed);
      server.listen(port, "127.0.0.1", () => {
        server.off("error", failed);
        resolve(undefined);
      });
    });
    if (error === undefined) break;
    if (error.code !== "EADDRINUSE" || performance.now() > until) throw error;
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
  }
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no server address");
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    received: (path) => requests.filter((request) => request.path === path),
    waitFor: async (predicate, timeoutMs = 5000) => {
      const deadline = performance.now() + timeoutMs;
      while (!predicate()) {
        if (performance.now() > deadline) throw new Error("timed out waiting for the server");
        await new Promise<void>((resolve) => setTimeout(resolve, 5));
      }
    },
    close: async () => {
      for (const timer of timers) clearTimeout(timer);
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
};

/** A URL on which nothing listens. */
export const closedUrl = async (): Promise<string> => {
  const server = await startServer();
  const url = server.url;
  await server.close();
  return url;
};

export const pause = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));
