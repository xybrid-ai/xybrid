const translations = {
  en: {
    tagline: 'Hybrid Cloud-Edge ML Inference Orchestrator',
    description:
      'Run ML models on-device or in the cloud with intelligent routing based on device capabilities. Built in Rust with Flutter bindings.',
    quickstart: 'Quickstart',
    index: 'Documentation index',
    full: 'Full documentation',
  },
  zh: {
    tagline: '混合云边缘 ML 推理编排器',
    description:
      '在设备端或云端运行 ML 模型，基于设备能力智能路由。使用 Rust 构建，支持 Flutter 绑定。',
    quickstart: '快速开始',
    index: '文档索引',
    full: '完整文档',
  },
};

export function homeContent(lang: string) {
  return lang === 'zh' ? translations.zh : translations.en;
}
