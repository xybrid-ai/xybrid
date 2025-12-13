import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col justify-center text-center flex-1 px-4">
      <h1 className="text-4xl font-bold mb-4">Xybrid</h1>
      <p className="text-xl text-muted-foreground mb-8">
        Hybrid Cloud-Edge ML Inference Orchestrator
      </p>
      <p className="text-lg mb-8 max-w-2xl mx-auto">
        Run ML models on-device or in the cloud with intelligent routing based on device capabilities.
        Built in Rust with Flutter bindings.
      </p>
      <div className="flex gap-4 justify-center">
        <Link
          href="/docs/overview"
          className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
        >
          Get Started
        </Link>
        <Link
          href="/docs/architecture"
          className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          Architecture
        </Link>
      </div>
    </div>
  );
}
