import Link from 'next/link';

const translations = {
  en: {
    tagline: 'Hybrid Cloud-Edge ML Inference Orchestrator',
    description:
      'Run ML models on-device or in the cloud with intelligent routing based on device capabilities. Built in Rust with Flutter bindings.',
    quickstart: 'Quickstart',
  },
  zh: {
    tagline: '混合云边缘 ML 推理编排器',
    description:
      '在设备端或云端运行 ML 模型，基于设备能力智能路由。使用 Rust 构建，支持 Flutter 绑定。',
    quickstart: '快速开始',
  },
};

export default async function HomePage({
  params,
}: {
  params: Promise<{ lang: string }>;
}) {
  const { lang } = await params;
  const t = translations[lang as keyof typeof translations] || translations.en;

  return (
    <div className="flex flex-col justify-center text-center flex-1 px-4">
      <h1 className="text-4xl font-bold mb-4">Xybrid</h1>
      <p className="text-xl text-muted-foreground mb-8">{t.tagline}</p>
      <p className="text-lg mb-8 max-w-2xl mx-auto">{t.description}</p>
      <div className="flex gap-4 justify-center">
        <Link
          href={`/${lang}/docs/quickstart`}
          className="inline-flex items-center justify-center rounded-md border border-input bg-background px-6 py-3 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          {t.quickstart}
        </Link>
      </div>
    </div>
  );
}
