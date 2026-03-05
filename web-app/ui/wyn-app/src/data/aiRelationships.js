// Static supply-chain relationship data for the AI Stack visualization.
// Edges flow bottom-up: Energy -> Chips -> Infrastructure -> Models -> Application
// Confidence: "high" = confirmed deal/public, "medium" = widely reported, "low" = inferred

export const LAYER_ORDER = [
  'Energy Layer',
  'Chips Layer',
  'Infrastructure',
  'Models',
  'Application',
];

export const LAYER_COLORS = {
  'Energy Layer':   '#f59e0b', // amber
  'Chips Layer':    '#ef4444', // red
  'Infrastructure': '#3b82f6', // blue
  'Models':         '#a855f7', // purple
  'Application':    '#22c55e', // green
};

export const CONFIDENCE_META = {
  high:   { label: 'Confirmed', color: '#22c55e' },
  medium: { label: 'Reported',  color: '#f59e0b' },
  low:    { label: 'Inferred',  color: '#ef4444' },
};

// Friendly names for tooltip display
export const TICKER_NAMES = {
  // Energy Layer
  NEE: 'NextEra Energy', ETN: 'Eaton Corp', SO: 'Southern Co', GEV: 'GE Vernova',
  DUK: 'Duke Energy', CEG: 'Constellation Energy', SRE: 'Sempra', AEP: 'American Electric Power',
  PCG: 'PG&E', D: 'Dominion Energy', VST: 'Vistra', EXC: 'Exelon',
  XEL: 'Xcel Energy', PEG: 'PSEG', ED: 'Consolidated Edison', WEC: 'WEC Energy',
  PPL: 'PPL Corp', TLN: 'Talen Energy',
  // Chips Layer (public)
  NVDA: 'NVIDIA', TSM: 'TSMC', AVGO: 'Broadcom', ASML: 'ASML', AMD: 'AMD',
  QCOM: 'Qualcomm', TXN: 'Texas Instruments', ARM: 'Arm Holdings', AMAT: 'Applied Materials',
  MU: 'Micron', ADI: 'Analog Devices', LRCX: 'Lam Research', INTC: 'Intel',
  KLAC: 'KLA Corp', SNPS: 'Synopsys', CDNS: 'Cadence', MRVL: 'Marvell', NXPI: 'NXP Semi',
  // Chips Layer (private)
  CEREBRAS: 'Cerebras Systems', GROQ: 'Groq', SAMBANOVA: 'SambaNova Systems',
  // Infrastructure (public)
  MSFT: 'Microsoft', AMZN: 'Amazon', GOOGL: 'Alphabet', CSCO: 'Cisco', IBM: 'IBM',
  ANET: 'Arista Networks', DELL: 'Dell', EQIX: 'Equinix', SNOW: 'Snowflake', DLR: 'Digital Realty',
  VRT: 'Vertiv', NET: 'Cloudflare', CDW: 'CDW', NTAP: 'NetApp', HPE: 'HPE',
  SMCI: 'Super Micro', AKAM: 'Akamai', FFIV: 'F5 Networks',
  // Infrastructure (public — additional)
  CRWV: 'CoreWeave',
  // Infrastructure (private)
  LAMBDA: 'Lambda Labs',
  // Models (public — already listed above: MSFT, GOOGL, META, AMZN, ORCL, IBM, AI, COHR)
  META: 'Meta', ORCL: 'Oracle', AI: 'C3.ai', COHR: 'Coherent',
  // Models (private)
  OPENAI: 'OpenAI', ANTHROPIC: 'Anthropic', XAI: 'xAI', MISTRAL: 'Mistral AI', COHERE: 'Cohere',
  // Application (public)
  CRM: 'Salesforce', ADBE: 'Adobe', NOW: 'ServiceNow', PLTR: 'Palantir', PANW: 'Palo Alto Networks',
  CRWD: 'CrowdStrike', WDAY: 'Workday', TTD: 'Trade Desk', TEAM: 'Atlassian',
  DDOG: 'Datadog', HUBS: 'HubSpot', VEEV: 'Veeva', ZS: 'Zscaler',
  MDB: 'MongoDB', MNDY: 'monday.com', OKTA: 'Okta', GTLB: 'GitLab', PATH: 'UiPath',
  S: 'SentinelOne', CFLT: 'Confluent', FICO: 'Fair Isaac (FICO)',
  // Application (private)
  DATABRICKS: 'Databricks', SCALEAI: 'Scale AI', FIGMA: 'Figma', CANVA: 'Canva',
};

// Private companies that appear only in the 3D graph (not in TradingView ticker list).
// Maps ticker key -> layer name.
export const PRIVATE_COMPANIES = {
  // Chips
  CEREBRAS:   'Chips Layer',
  GROQ:       'Chips Layer',
  SAMBANOVA:  'Chips Layer',
  // Infrastructure
  LAMBDA:     'Infrastructure',
  // Models
  OPENAI:     'Models',
  ANTHROPIC:  'Models',
  XAI:        'Models',
  MISTRAL:    'Models',
  COHERE:     'Models',
  // Application
  DATABRICKS: 'Application',
  SCALEAI:    'Application',
  FIGMA:      'Application',
  CANVA:      'Application',
};

// Supply-chain edges with confidence levels
export const EDGES = [
  // ══════════════════════════════════════
  // Energy Layer -> Chips Layer
  // ══════════════════════════════════════
  { from: 'CEG', to: 'INTC',  reason: 'Nuclear power for Intel fabs',           confidence: 'medium' },
  { from: 'CEG', to: 'NVDA',  reason: 'Clean energy for NVIDIA HQ',             confidence: 'low' },
  { from: 'CEG', to: 'TSM',   reason: 'Energy for TSMC Arizona fab',            confidence: 'medium' },
  { from: 'VST', to: 'TXN',   reason: 'Texas grid power for TI fabs',           confidence: 'medium' },
  { from: 'VST', to: 'NVDA',  reason: 'Texas power for NVIDIA operations',      confidence: 'low' },
  { from: 'NEE', to: 'NVDA',  reason: 'Renewable energy contracts',             confidence: 'low' },
  { from: 'NEE', to: 'AMD',   reason: 'Renewable energy for AMD facilities',    confidence: 'low' },
  { from: 'TLN', to: 'AMZN',  reason: 'Talen-Amazon data center power deal',    confidence: 'high' },
  { from: 'DUK', to: 'AVGO',  reason: 'Southeast grid power',                   confidence: 'low' },
  { from: 'SO',  to: 'INTC',  reason: 'Southern grid for Intel operations',     confidence: 'low' },
  { from: 'ED',  to: 'IBM',   reason: 'NYC grid power for IBM facilities',      confidence: 'medium' },
  { from: 'PEG', to: 'ASML',  reason: 'NJ power for ASML US operations',        confidence: 'low' },
  { from: 'AEP', to: 'MU',    reason: 'Grid power for Micron fabs',             confidence: 'low' },
  { from: 'EXC', to: 'QCOM',  reason: 'Grid power for Qualcomm',               confidence: 'low' },
  { from: 'SRE', to: 'QCOM',  reason: 'SoCal power for Qualcomm HQ',           confidence: 'medium' },
  { from: 'PCG', to: 'AMD',   reason: 'California power for AMD',               confidence: 'medium' },
  { from: 'XEL', to: 'ADI',   reason: 'Grid power for Analog Devices',          confidence: 'low' },
  { from: 'GEV', to: 'TSM',   reason: 'Power equipment for fab infrastructure', confidence: 'medium' },
  { from: 'ETN', to: 'TSM',   reason: 'Power management for chip fabs',         confidence: 'high' },
  { from: 'ETN', to: 'INTC',  reason: 'Power distribution for Intel fabs',      confidence: 'high' },
  { from: 'D',   to: 'MRVL',  reason: 'Virginia power for Marvell',             confidence: 'low' },
  { from: 'PPL', to: 'ARM',   reason: 'Grid power for Arm operations',          confidence: 'low' },
  { from: 'WEC', to: 'NXPI',  reason: 'Midwest power for NXP',                  confidence: 'low' },
  // Energy -> private chips
  { from: 'CEG', to: 'CEREBRAS', reason: 'Nuclear power for Cerebras operations', confidence: 'low' },

  // ══════════════════════════════════════
  // Chips Layer -> Infrastructure
  // ══════════════════════════════════════
  { from: 'NVDA', to: 'MSFT',     reason: 'GPUs for Azure AI cloud',            confidence: 'high' },
  { from: 'NVDA', to: 'AMZN',     reason: 'GPUs for AWS data centers',          confidence: 'high' },
  { from: 'NVDA', to: 'GOOGL',    reason: 'GPUs for Google Cloud',              confidence: 'high' },
  { from: 'NVDA', to: 'DELL',     reason: 'GPUs in Dell AI servers',            confidence: 'high' },
  { from: 'NVDA', to: 'HPE',      reason: 'GPUs in HPE servers',                confidence: 'high' },
  { from: 'NVDA', to: 'SMCI',     reason: 'GPUs in SuperMicro AI servers',      confidence: 'high' },
  { from: 'NVDA', to: 'CRWV', reason: 'Largest GPU cloud customer',        confidence: 'high' },
  { from: 'NVDA', to: 'LAMBDA',   reason: 'GPUs for Lambda cloud',              confidence: 'high' },
  { from: 'AMD',  to: 'MSFT',     reason: 'EPYC CPUs for Azure',               confidence: 'high' },
  { from: 'AMD',  to: 'AMZN',     reason: 'EPYC CPUs for AWS',                 confidence: 'high' },
  { from: 'AMD',  to: 'GOOGL',    reason: 'CPUs for Google Cloud',              confidence: 'high' },
  { from: 'AMD',  to: 'DELL',     reason: 'CPUs in Dell servers',               confidence: 'high' },
  { from: 'AMD',  to: 'HPE',      reason: 'CPUs in HPE servers',                confidence: 'high' },
  { from: 'INTC', to: 'MSFT',     reason: 'Xeon CPUs for Azure',               confidence: 'high' },
  { from: 'INTC', to: 'DELL',     reason: 'Xeon CPUs in Dell servers',          confidence: 'high' },
  { from: 'INTC', to: 'IBM',      reason: 'Processors for IBM systems',         confidence: 'medium' },
  { from: 'TSM',  to: 'NVDA',     reason: 'Fabricates NVIDIA chips',            confidence: 'high' },
  { from: 'TSM',  to: 'AMD',      reason: 'Fabricates AMD chips',               confidence: 'high' },
  { from: 'TSM',  to: 'AVGO',     reason: 'Fabricates Broadcom chips',          confidence: 'high' },
  { from: 'TSM',  to: 'QCOM',     reason: 'Fabricates Qualcomm chips',          confidence: 'high' },
  { from: 'TSM',  to: 'CEREBRAS', reason: 'Fabricates Cerebras wafer chips',    confidence: 'high' },
  { from: 'AVGO', to: 'GOOGL',    reason: 'Custom TPU networking chips',        confidence: 'high' },
  { from: 'AVGO', to: 'CSCO',     reason: 'Networking silicon for Cisco',       confidence: 'high' },
  { from: 'AVGO', to: 'MSFT',     reason: 'Custom AI accelerators',             confidence: 'medium' },
  { from: 'MU',   to: 'NVDA',     reason: 'HBM memory for GPUs',               confidence: 'high' },
  { from: 'MU',   to: 'SMCI',     reason: 'Memory for AI servers',              confidence: 'medium' },
  { from: 'MRVL', to: 'AMZN',     reason: 'Custom cloud chips for AWS',        confidence: 'high' },
  { from: 'MRVL', to: 'MSFT',     reason: 'Custom silicon for Azure',           confidence: 'medium' },
  { from: 'ARM',  to: 'AMZN',     reason: 'Architecture for AWS Graviton',      confidence: 'high' },
  { from: 'ARM',  to: 'GOOGL',    reason: 'Architecture for Google Axion',      confidence: 'high' },
  { from: 'ANET', to: 'MSFT',     reason: 'Network switches for Azure',         confidence: 'high' },
  { from: 'CSCO', to: 'AMZN',     reason: 'Networking for AWS',                 confidence: 'medium' },
  { from: 'AMAT', to: 'TSM',      reason: 'Fab equipment for TSMC',             confidence: 'high' },
  { from: 'LRCX', to: 'TSM',      reason: 'Etch equipment for TSMC',            confidence: 'high' },
  { from: 'KLAC', to: 'TSM',      reason: 'Inspection equipment for TSMC',      confidence: 'high' },
  { from: 'ASML', to: 'TSM',      reason: 'EUV lithography for TSMC',           confidence: 'high' },
  { from: 'ASML', to: 'INTC',     reason: 'EUV lithography for Intel',          confidence: 'high' },
  { from: 'SNPS', to: 'NVDA',     reason: 'Chip design tools for NVIDIA',       confidence: 'high' },
  { from: 'CDNS', to: 'AMD',      reason: 'Chip design tools for AMD',          confidence: 'high' },
  { from: 'NXPI', to: 'CSCO',     reason: 'Networking chips for Cisco',         confidence: 'medium' },
  { from: 'ADI',  to: 'VRT',      reason: 'Power ICs for Vertiv systems',       confidence: 'medium' },
  { from: 'QCOM', to: 'MSFT',     reason: 'AI chips for Copilot+ PCs',         confidence: 'high' },
  { from: 'TXN',  to: 'DELL',     reason: 'Analog chips in Dell systems',       confidence: 'medium' },
  // Private chips -> infrastructure
  { from: 'CEREBRAS', to: 'CRWV', reason: 'Cerebras CS-3 on CoreWeave cloud', confidence: 'high' },
  { from: 'GROQ',     to: 'GROQ',      reason: 'Groq LPU inference cloud (self)', confidence: 'high' },

  // ══════════════════════════════════════
  // Infrastructure -> Models
  // ══════════════════════════════════════
  { from: 'MSFT',     to: 'OPENAI',    reason: 'Azure hosts all OpenAI training',         confidence: 'high' },
  { from: 'MSFT',     to: 'MISTRAL',   reason: 'Azure partnership with Mistral',          confidence: 'high' },
  { from: 'AMZN',     to: 'ANTHROPIC', reason: 'AWS $4B investment in Anthropic',         confidence: 'high' },
  { from: 'AMZN',     to: 'COHERE',    reason: 'AWS Bedrock hosts Cohere models',         confidence: 'high' },
  { from: 'GOOGL',    to: 'GOOGL',     reason: 'GCP trains Gemini models',                confidence: 'high' },
  { from: 'GOOGL',    to: 'ANTHROPIC', reason: 'Google $2B investment in Anthropic',      confidence: 'high' },
  { from: 'MSFT',     to: 'META',      reason: 'Azure compute for Llama training',        confidence: 'medium' },
  { from: 'AMZN',     to: 'META',      reason: 'AWS for Meta AI workloads',               confidence: 'medium' },
  { from: 'ORCL',     to: 'OPENAI',    reason: 'OCI for OpenAI overflow training',        confidence: 'high' },
  { from: 'ORCL',     to: 'XAI',       reason: 'OCI hosts xAI Grok training',             confidence: 'high' },
  { from: 'ORCL',     to: 'COHERE',    reason: 'OCI partnership with Cohere',             confidence: 'medium' },
  { from: 'IBM',      to: 'IBM',       reason: 'IBM Cloud trains watsonx',                confidence: 'high' },
  { from: 'CRWV', to: 'OPENAI',   reason: 'CoreWeave GPU cloud for OpenAI',          confidence: 'high' },
  { from: 'CRWV', to: 'MISTRAL',  reason: 'CoreWeave trains Mistral models',         confidence: 'medium' },
  { from: 'EQIX',     to: 'META',      reason: 'Colocation for Meta AI clusters',         confidence: 'medium' },
  { from: 'DLR',      to: 'MSFT',      reason: 'Data center space for Azure AI',          confidence: 'medium' },
  { from: 'DLR',      to: 'ORCL',      reason: 'Data center space for OCI',               confidence: 'medium' },
  { from: 'SMCI',     to: 'META',      reason: 'AI server racks for Llama training',      confidence: 'medium' },
  { from: 'DELL',     to: 'ORCL',      reason: 'Servers for Oracle Cloud AI',             confidence: 'medium' },
  { from: 'DELL',     to: 'XAI',       reason: 'Dell servers for xAI Colossus cluster',   confidence: 'high' },
  { from: 'VRT',      to: 'AMZN',      reason: 'Cooling systems for AWS DCs',             confidence: 'medium' },
  { from: 'VRT',      to: 'GOOGL',     reason: 'Power/cooling for Google DCs',            confidence: 'medium' },
  { from: 'SNOW',     to: 'AI',        reason: 'Data platform for C3.ai',                 confidence: 'low' },
  { from: 'NET',      to: 'AI',        reason: 'Edge inference for C3.ai',                confidence: 'low' },
  { from: 'AMZN',     to: 'COHR',      reason: 'Cloud for Coherent AI workloads',         confidence: 'low' },
  { from: 'LAMBDA',   to: 'MISTRAL',   reason: 'Lambda GPU cloud for Mistral training',   confidence: 'medium' },

  // ══════════════════════════════════════
  // Models -> Application
  // ══════════════════════════════════════
  // OpenAI -> apps
  { from: 'OPENAI', to: 'CRM',        reason: 'GPT powers Salesforce Einstein GPT',       confidence: 'high' },
  { from: 'OPENAI', to: 'ADBE',       reason: 'GPT powers Adobe Firefly text features',   confidence: 'medium' },
  { from: 'OPENAI', to: 'NOW',        reason: 'GPT integration in ServiceNow',            confidence: 'high' },
  { from: 'OPENAI', to: 'TEAM',       reason: 'GPT for Atlassian Intelligence',           confidence: 'medium' },
  { from: 'OPENAI', to: 'HUBS',       reason: 'GPT for HubSpot AI features',              confidence: 'medium' },
  { from: 'OPENAI', to: 'WDAY',       reason: 'GPT for Workday AI assistant',             confidence: 'medium' },
  { from: 'OPENAI', to: 'OKTA',       reason: 'AI-powered identity threat detection',     confidence: 'low' },
  { from: 'OPENAI', to: 'ZS',         reason: 'AI-powered zero trust security',           confidence: 'low' },
  { from: 'OPENAI', to: 'MNDY',       reason: 'GPT for monday.com AI assistant',          confidence: 'medium' },
  { from: 'OPENAI', to: 'VEEV',       reason: 'GPT for Veeva CRM AI',                    confidence: 'low' },
  { from: 'OPENAI', to: 'FIGMA',      reason: 'GPT powers Figma AI design features',     confidence: 'medium' },
  { from: 'OPENAI', to: 'CANVA',      reason: 'GPT powers Canva Magic Write',            confidence: 'high' },
  { from: 'OPENAI', to: 'DATABRICKS', reason: 'GPT integration in Databricks',           confidence: 'medium' },
  // Anthropic -> apps
  { from: 'ANTHROPIC', to: 'PLTR',    reason: 'Claude models in Palantir AIP',            confidence: 'high' },
  { from: 'ANTHROPIC', to: 'NOW',     reason: 'Claude for ServiceNow workflows',          confidence: 'medium' },
  { from: 'ANTHROPIC', to: 'DDOG',    reason: 'Claude for Datadog AI operations',         confidence: 'low' },
  { from: 'ANTHROPIC', to: 'GTLB',    reason: 'Claude for GitLab Duo Code',              confidence: 'medium' },
  { from: 'ANTHROPIC', to: 'DATABRICKS', reason: 'Claude available on Databricks',       confidence: 'high' },
  { from: 'ANTHROPIC', to: 'SCALEAI', reason: 'Scale AI uses Claude for data ops',        confidence: 'medium' },
  // Google / Gemini -> apps
  { from: 'GOOGL', to: 'CRM',         reason: 'Vertex AI for Salesforce',                 confidence: 'medium' },
  { from: 'GOOGL', to: 'PLTR',        reason: 'GCP partnership with Palantir',            confidence: 'high' },
  { from: 'GOOGL', to: 'DDOG',        reason: 'Gemini for Datadog AI ops',                confidence: 'low' },
  { from: 'GOOGL', to: 'PANW',        reason: 'AI for Palo Alto security',                confidence: 'medium' },
  { from: 'GOOGL', to: 'S',           reason: 'AI for SentinelOne security',              confidence: 'low' },
  // Meta / Llama -> apps (open source)
  { from: 'META',  to: 'PLTR',        reason: 'Llama models in Palantir',                 confidence: 'high' },
  { from: 'META',  to: 'CRWD',        reason: 'Llama for CrowdStrike security AI',        confidence: 'medium' },
  { from: 'META',  to: 'GTLB',        reason: 'Llama for GitLab Duo',                     confidence: 'medium' },
  { from: 'META',  to: 'PATH',        reason: 'Llama for UiPath automation',              confidence: 'medium' },
  { from: 'META',  to: 'DATABRICKS',  reason: 'Llama fine-tuning on Databricks',          confidence: 'high' },
  // Mistral -> apps
  { from: 'MISTRAL', to: 'CRM',       reason: 'Mistral models in Salesforce',             confidence: 'medium' },
  { from: 'MISTRAL', to: 'DATABRICKS', reason: 'Mistral available on Databricks',         confidence: 'high' },
  // Cohere -> apps
  { from: 'COHERE', to: 'SCALEAI',    reason: 'Cohere enterprise models for Scale',       confidence: 'low' },
  { from: 'COHERE', to: 'DATABRICKS', reason: 'Cohere on Databricks marketplace',         confidence: 'medium' },
  // Amazon Bedrock -> apps
  { from: 'AMZN',  to: 'PLTR',        reason: 'AWS Bedrock for Palantir AIP',             confidence: 'high' },
  { from: 'AMZN',  to: 'CRWD',        reason: 'Bedrock AI for CrowdStrike',               confidence: 'medium' },
  { from: 'AMZN',  to: 'DDOG',        reason: 'AWS AI for Datadog',                       confidence: 'medium' },
  { from: 'AMZN',  to: 'MDB',         reason: 'AWS AI for MongoDB Atlas',                 confidence: 'medium' },
  { from: 'AMZN',  to: 'CFLT',        reason: 'AWS for Confluent streaming AI',           confidence: 'medium' },
  { from: 'AMZN',  to: 'TTD',         reason: 'AWS AI for Trade Desk',                    confidence: 'low' },
  { from: 'AMZN',  to: 'NET',         reason: 'Cloud AI for Cloudflare',                  confidence: 'low' },
  { from: 'AMZN',  to: 'SNOW',        reason: 'AWS for Snowflake AI features',            confidence: 'medium' },
  // Other model -> app connections
  { from: 'ORCL',  to: 'CRM',         reason: 'OCI AI for Salesforce',                    confidence: 'medium' },
  { from: 'IBM',   to: 'NOW',         reason: 'watsonx for ServiceNow',                   confidence: 'medium' },
  { from: 'IBM',   to: 'PANW',        reason: 'AI security with Palo Alto',               confidence: 'medium' },
  { from: 'AI',    to: 'PLTR',        reason: 'C3.ai enterprise AI platform',             confidence: 'low' },
  { from: 'COHR',  to: 'NET',         reason: 'Optical networking for Cloudflare',        confidence: 'medium' },
  { from: 'MSFT',  to: 'ADBE',        reason: 'Copilot integration with Adobe',           confidence: 'high' },
  // Anthropic -> FICO
  { from: 'ANTHROPIC', to: 'FICO',    reason: 'Claude AI powers FICO fraud/credit decisioning', confidence: 'high' },
  // Amazon -> FICO
  { from: 'AMZN',  to: 'FICO',        reason: 'AWS cloud infrastructure for FICO Platform',     confidence: 'high' },
  // Other -> FICO
  { from: 'MSFT',  to: 'FICO',        reason: 'Azure AI services for FICO analytics',           confidence: 'medium' },
  { from: 'ORCL',  to: 'FICO',        reason: 'Oracle databases for FICO data processing',      confidence: 'medium' },
  // xAI -> apps
  { from: 'XAI',   to: 'SCALEAI',     reason: 'Scale AI provides data for xAI/Grok',     confidence: 'medium' },
  // Scale AI -> model training support
  { from: 'SCALEAI', to: 'OPENAI',    reason: 'Scale AI provides RLHF data for OpenAI',  confidence: 'high' },
  { from: 'SCALEAI', to: 'ANTHROPIC', reason: 'Scale AI provides eval data for Anthropic', confidence: 'medium' },
  { from: 'SCALEAI', to: 'META',      reason: 'Scale AI data labeling for Llama',         confidence: 'high' },
];
