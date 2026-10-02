# HeritAgent  
## Causally-Gated Assimilation of Acquired Skills into Functional Genomes for Evolvable LLM Agents

### Master Research & Engineering Specification — v0.1

## 1. Projenin amacı

HeritAgent, Large Language Model tabanlı agent sistemlerinin yalnızca tek oturum içinde öğrenmesini veya basit biçimde geçmiş deneyimlerini hatırlamasını değil, **birden fazla nesil boyunca yapısal olarak gelişmesini** araştıran model-bağımsız bir agent evolution framework'üdür.

Projenin temel fikri biyolojik evrimin doğrudan taklidini yapmak değildir. Biyolojideki bazı güçlü kavramları — varyasyon, phenotype, genotype, çevresel baskı, yapay seçilim, kalıtım, yaşam boyu adaptasyon ve genetic assimilation — agent sistemleri için açık, ölçülebilir ve çalıştırılabilir mühendislik yapılarına dönüştürmektir.

HeritAgent'ın merkezindeki araştırma sorusu şudur:

> Bir agentın yaşamı sırasında öğrendiği yeni bir fonksiyonel davranış veya skill, yalnızca belleğe kaydedilmek yerine bağımsız deneylerle faydası doğrulandıktan sonra gelecek agent nesillerinin kalıtsal mimarisinin bir parçası haline getirilebilir mi?

Bu nedenle HeritAgent klasik bir memory sistemi, prompt optimizer, genetic algorithm veya agent architecture search framework'ü değildir. Bunların bazı mekanizmalarını kullanabilir ancak temel katkısı **lifetime-acquired capability → causal validation → heritable functional genome** dönüşümüdür.

---

# 2. Ana araştırma tezi

HeritAgent'ın merkezi hipotezi:

> **Causal Genetic Assimilation Hypothesis for Agents**

şeklinde tanımlanacaktır.

Hipotez:

> LLM agentlarının görev sırasında edindiği faydalı yetenekler, bu yeteneklerin görev performansına nedensel katkısı kontrollü biçimde doğrulandıktan sonra kalıtsal bir functional agent genome'a asimile edilebilir. Bu süreç tekrarlandığında sonraki nesiller, foundation model ağırlıkları değiştirilmeden dahi, başlangıç popülasyonundan daha yüksek zero-shot performans, daha düşük adaptasyon maliyeti ve daha iyi genelleme gösterebilir.

Bunun anlamı şudur:

```text
Generation G0
      │
      ▼
Environment
      │
      ▼
Experience
      │
      ▼
Lifetime Learning
      │
      ▼
Acquired Trait
      │
      ▼
Causal Validation
      │
      ├── fail ──► discard / somatic memory
      │
      └── pass
             │
             ▼
       Genetic Assimilation
             │
             ▼
        Genome G1
             │
             ▼
       Generation G1
             │
            ...
```

Foundation model bu sürecin tamamında frozen olabilir.

Dolayısıyla gelişen temel olarak model ağırlığı değil, **agent organizmasıdır**.

---

# 3. Literatürde konumumuz

HeritAgent'ı doğru konumlandırmak için hangi fikirlerin artık özgün olmadığını baştan kabul etmeliyiz.

AgentSquare, agent tasarımını planning, reasoning, tool use ve memory modüllerine ayırıyor; bu modüller üzerinde evolution ve recombination uyguluyor. Dolayısıyla “farklı agentların iyi modüllerini birleştirme” fikri tek başına bizim contribution'ımız olamaz. AgentSquare ICLR 2025'te altı benchmark üzerinde insan tasarımı agentlara karşı ortalama %17,2 performans artışı raporladı.

Darwin Gödel Machine, frozen foundation modeller üzerinde çalışan coding agentlarını kendi kodlarını değiştirmeye teşvik ediyor ve başarılı/ilginç agentlardan büyüyen bir archive/lineage oluşturuyor. Bu nedenle “nesiller, soy ağacı ve açık uçlu agent evolution” kavramları da tek başına yeni değildir.

Mendel Gödel Machine daha da yakına gelir: reaction-norm mutation ve cross-lineage hybridization mekanizmalarıyla farklı agent soylarından birikmiş evidence'ı karşılaştırmalı biçimde kullanır ve bunu controlled inheritance bağlamında sunar. Bu nedenle cross-lineage inheritance veya crossover ana novelty olarak sunulmamalıdır.

Genomebook, LLM agentları için doğrudan genotype/phenotype dili kullanır; 26 behavioural trait'i 60 diploid locus üzerinden sekiz nesil boyunca kalıtır. Ancak burada genotype büyük ölçüde behavioural/personality özelliklerini parameterize etmektedir. HeritAgent'ın hedefi ise **çalıştırılabilir fonksiyonel capability'lerin** genotype haline gelmesidir: planning policy, memory strategy, tool policy, verification logic, workflow, skill ve benzeri agent operasyonları.

SkillRL başarılı ve başarısız trajectory'lerden high-level reusable skills çıkarır, bunları SkillBank içinde organize eder ve skill library'nin policy ile recursive biçimde evolve olmasını sağlar. Dolayısıyla “trajectory → reusable skill → future reuse” de yeni değildir.

CapaBench/ShapleyFlow agent workflow bileşenlerinin performansa katkısını Shapley values kullanarak ölçer. SkillShapley ise agent skills içindeki tek tek adımların marjinal katkısını değerlendirir. Dolayısıyla causal/contribution attribution kavramı da başlı başına özgün değildir.

HeritAgent'ın hedeflediği özgün kombinasyon şu nedenle daha dardır:

```text
Lifetime-acquired capability
        +
explicit somatic state
        +
executable functional genome
        +
controlled replay / ablation
        +
causal contribution estimation
        +
held-out generalization validation
        +
inheritance decision
        +
cross-generational assimilation
```

Mevcut literatür taramasında bu tam zinciri tek bir modern LLM agent framework'ünde birlikte uygulayan doğrudan eşdeğer bir çalışma tespit edilmemiştir. Bu yine de kesin bir “ilk çalışma” iddiası değildir; proje boyunca devam eden literature monitoring ile doğrulanmalıdır.

---

# 4. HeritAgent'ın biyolojik analojisi

Biyolojik terminoloji yalnızca metafor olarak kullanılmayacaktır. Her kavramın computational bir karşılığı bulunacaktır.

| Biyolojik kavram | HeritAgent karşılığı |
|---|---|
| Genome / genotype | Kalıtsal executable agent specification |
| Gene | Ayrı değiştirilebilir functional agent trait |
| Regulatory gene | Trait'in ne zaman aktive olacağını belirleyen koşullar |
| Phenotype | Gerçek task sırasında ortaya çıkan execution behaviour |
| Environment | Benchmark veya görev dağılımı |
| Fitness | Çok boyutlu agent performansı |
| Mutation | Genome üzerinde yeni varyasyon |
| Recombination | Birden fazla lineage'dan functional trait kombinasyonu |
| Selection | Hangi genome/trait'lerin devam edeceğinin seçilmesi |
| Lifetime learning | Agentın kendi task deneyimi sırasında yeni capability kazanması |
| Somatic adaptation | Lifetime içinde öğrenilmiş ama henüz kalıtsal olmayan trait |
| Germline | Nesiller arasında taşınabilen doğrulanmış genome |
| Genetic assimilation | Acquired trait'in kalıtsal genome içine terfi ettirilmesi |
| Population | Aynı generation içindeki farklı genome'lara sahip agentlar |
| Lineage | Agentların parent/child evrimsel ilişkileri |

HeritAgent'ta biyolojik analoji hiçbir zaman fiziksel DNA veya gerçek genetik mekanizmanın birebir simülasyonu olarak sunulmayacaktır. Amaç, evrimsel bilgi aktarımını **yorumlanabilir ve deneysel olarak ölçülebilir agent architecture primitive'lerine** dönüştürmektir.

---

# 5. Sistemin temel ayrımı: Germline ve Somatic State

HeritAgent'ın en önemli architectural distinction'ı budur.

Bir agent iki ayrı bilgi alanıyla çalışır:

```text
                    AGENT
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
      GERMLINE               SOMATIC
       GENOME                  STATE
           │                     │
    inherited traits        lifetime learning
    validated skills        temporary strategies
    policies                task discoveries
    regulatory logic        experimental skills
```

**Germline**, agent doğduğunda aldığı ve gelecek nesillere aktarılabilecek functional genome'dur.

**Somatic state**, agentın görev sırasında öğrendiği şeylerdir. Bunlar agentın kendi yaşamı sırasında kullanılabilir ancak otomatik olarak kalıtsal değildir.

Bu ayrım HeritAgent için zorunludur.

Aksi halde sistem:

```text
bir şey öğrendi
→ memory'ye yazdı
→ gelecek agent kullandı
```

seviyesinde kalır ve SkillRL gibi persistent-skill sistemlerinden kavramsal olarak yeterince ayrışmaz.

HeritAgent:

```text
learn
→ candidate trait
→ validate
→ attribute
→ generalize
→ assimilate
→ inherit
```

yapacaktır.

---

# 6. Executable Agent Genome — EAG

HeritAgent'ın temel veri yapısı **Executable Agent Genome (EAG)** olacaktır.

EAG yalnızca prompt metni değildir.

Foundation modelin üzerinde çalışan agentın fonksiyonel mimarisini tanımlayan typed, versioned ve executable bir specification'dır.

Örnek:

```yaml
genome:
  id: G-00421
  generation: 17

  cognition:
    planner: hierarchical_v5
    hypothesis_policy: branch_and_prune_v2
    verifier: evidence_first_v4
    reflection: selective_v3

  memory:
    episodic: graph_memory_v3
    retrieval: salience_v6
    compression: semantic_delta_v2

  execution:
    tool_selector: adaptive_v7
    retry_policy: hypothesis_shift_v4
    error_recovery: causal_recovery_v2

  organization:
    topology: planner_executor_critic
    delegation: confidence_routing_v3

  skills:
    - repository_debugging_v12
    - constraint_recovery_v4
    - source_synthesis_v8

  regulation:
    repository_debugging_v12:
      express_when:
        failing_tests_gte: 2

    constraint_recovery_v4:
      express_when:
        repeated_failure_gte: 2
```

Bu genome farklı foundation modellere expression layer üzerinden uygulanabilir.

Ama genotype'ın model-independent olması bir varsayım değil, ayrıca test edilecek research hypothesis'tir.

---

# 7. Genotype → phenotype

Agent performance yalnızca genome tarafından belirlenmez.

Formel olarak:

\[
P = \Phi(G, M, E, B)
\]

burada:

- \(G\): Executable Agent Genome,
- \(M\): foundation model,
- \(E\): environment/task distribution,
- \(B\): execution budget,
- \(P\): ortaya çıkan phenotype.

Phenotype bir skor değil, agentın gerçek davranışıdır:

```text
observations
decisions
tool calls
errors
recoveries
plans
replans
latency
token usage
task result
```

Bu nedenle HeritAgent'ın trajectory recorder'ı yalnızca final reward değil, agentın tüm functional phenotype'ını kaydetmelidir.

---

# 8. Lifetime Learning Engine

Her agent kendi lifetime'ı boyunca deneyimlerden yeni davranışlar öğrenebilir.

Örneğin bir coding task'ta:

```text
attempt
↓
failure
↓
diagnosis
↓
new strategy
↓
successful resolution
```

ortaya çıkabilir.

Trait Miner bu trajectory'den potansiyel olarak tekrar kullanılabilir bir capability çıkarır:

```yaml
candidate_trait:
  id: CT-819
  type: skill
  source_trajectory: T-18421
  purpose: failure_recovery
  scope: repository_debugging
  estimated_generality: medium
```

Bu candidate doğrudan genome'a yazılmaz.

**Somatic Trait Store** içerisine alınır.

---

# 9. Causal Inheritance Gate — CIG

HeritAgent'ın merkezi research mechanism'ı **Causal Inheritance Gate** olacaktır.

CIG şu soruyu cevaplar:

> Agentın task sırasında edindiği bu trait gerçekten başarılı sonuca neden oldu mu ve gelecek agentlara aktarılmaya değer mi?

Bir trait \(t\), genome \(G\) ve environment distribution \(E\) için temel katkı:

\[
C(t|G)=
\mathbb E_{e\sim E}
[
F(G\cup t,e)-F(G\setminus t,e)
]
\]

olarak düşünülebilir.

Pratik sistem tek bir ölçüme güvenmemelidir.

CIG dört aşamalı çalışacaktır:

```text
Candidate Trait
      │
      ▼
1. Replay Validation
      │
      ▼
2. Controlled Ablation
      │
      ▼
3. Held-out Generalization
      │
      ▼
4. Interaction / Regression Tests
      │
      ▼
Contribution Estimate
      │
   ┌──┴──┐
 reject  promote
          │
          ▼
 Genetic Assimilation
```

Replay, trait'in keşfedildiği environment üzerinde gerçekten reproducible olup olmadığını test eder.

Ablation, trait kaldırıldığında performance'ın ne kadar değiştiğini ölçer.

Generalization, trait'in yalnızca keşfedildiği örneği ezberlemediğini göstermek için görülmemiş benzer görevlerde test eder.

Interaction testing, trait'in genome'daki diğer functional genes ile zararlı interaction oluşturup oluşturmadığını ölçer.

Shapley-style attribution veya daha ucuz approximation yöntemleri özellikle interaction bulunan genome'larda kullanılabilir. Attribution literatüründe ShapleyFlow ve SkillShapley bu tür component/skill contribution analizlerinin uygulanabilir olduğunu göstermektedir.

---

# 10. Validated Trait Assimilation — VTA

CIG'i geçen candidate trait için **Validated Trait Assimilation** uygulanır.

Bir trait'in kalıtsal hale gelebilmesi için örneğin:

\[
Contribution(t)>\tau_c
\]

\[
Generalization(t)>\tau_g
\]

\[
RegressionRisk(t)<\tau_r
\]

\[
Cost(t)<\tau_k
\]

koşulları aranabilir.

Threshold'lar sabit kalmayabilir; environment veya generation'a göre adaptive olabilir.

Assimilation sonrasında trait:

```text
Somatic Trait
      ↓
CIG PASS
      ↓
Validated Trait
      ↓
Germline Registry
      ↓
Executable Agent Genome
```

haline gelir.

Bu mekanizma HeritAgent'ın ana novelty claim'inin çekirdeğidir.

---

# 11. Evolution Engine

Genetic assimilation tek başına yeterli değildir. Gerçek bir multi-generational evolution sistemi için population dynamics de gerekir.

Her generation:

```text
Population Gt
     │
     ▼
Task Distribution
     │
     ▼
Phenotypes
     │
     ▼
Fitness + Diversity + Cost
     │
     ▼
Selection
     │
 ┌───┼────────────┐
 │   │            │
elite mutation recombination
 │   │            │
 └───┼────────────┘
     ▼
Population Gt+1
```

Mutation üç biçimde uygulanabilir.

Random mutation küçük genome/config değişiklikleri üretir.

LLM-guided mutation başarısız trajectory veya fitness feedback'i analiz ederek hedefli genome varyasyonları üretir.

Evidence-guided mutation belirli tekrar eden failure pattern'lerini düzeltecek functional genes önerir.

Recombination birden fazla lineage'dan functional traits birleştirebilir. Ancak Mendel Gödel Machine ve AgentSquare nedeniyle mutation/recombination HeritAgent'ın temel novelty claim'i değildir; engine'in destekleyici mekanizmalarıdır.

---

# 12. Population ve diversity

HeritAgent tek bir “en iyi agent” aramamalıdır.

Aşırı selection pressure population collapse oluşturabilir.

Bu nedenle population manager yalnızca en yüksek fitness'i değil:

```text
fitness
diversity
novelty
specialization
cost
lineage potential
```

gibi faktörleri dengelemelidir.

Bir agent düşük current fitness'e rağmen gelecekte yararlı bir lineage'ın başlangıcı olabilir. Darwin/Huxley-Gödel yaklaşımındaki archive düşüncesi burada related-work/reference olarak kullanılacaktır; HeritAgent bunu kendi novelty'si olarak sunmayacaktır.

Her generation içinde birden fazla ecological niche tutulabilir:

```text
Population
├── coding specialists
├── tool-use specialists
├── planning specialists
└── generalists
```

İlk paper için niche mekanizması zorunlu değildir; architecture tarafından desteklenmesi yeterlidir.

---

# 13. Lineage Graph

Her agent ve trait'in soy ilişkisi açıkça tutulmalıdır.

Örnek:

```text
G0:A0
├── G1:A3
│   ├── G2:A8
│   └── G2:A9
│       └── G3:A21
│
└── G1:A4
    └── G2:A11
```

Parent-child edge yalnızca soy ilişkisi içermemelidir.

```json
{
  "parent": "A9",
  "child": "A21",

  "inheritance": [
    "planner_v5",
    "memory_v3"
  ],

  "assimilated_traits": [
    "recovery_skill_17"
  ],

  "mutations": [
    "tool_policy_v4 -> tool_policy_v5"
  ],

  "fitness_delta": 0.084
}
```

Lineage Store hem analysis hem reproducibility açısından first-class component olacaktır.

---

# 14. Fitness modeli

Tek scalar task success yeterli değildir.

HeritAgent multi-objective evaluation kullanmalıdır.

Temel fitness vector:

\[
F =
(
S,
R,
E,
G,
N,
-C,
-L
)
\]

burada:

- \(S\): task success,
- \(R\): reliability,
- \(E\): efficiency,
- \(G\): generalization,
- \(N\): novelty/diversity,
- \(C\): compute/token cost,
- \(L\): latency veya unnecessary actions.

Scalar selection gerektiğinde:

\[
F_s =
w_sS+
w_rR+
w_eE+
w_gG+
w_nN-
w_cC-
w_lL
\]

kullanılabilir.

Ancak ham component skorları her zaman saklanmalıdır.

---

# 15. Ana araştırma hipotezleri

Paper'ın deneyleri en az şu hipotezleri test edecektir.

**H1 — Assimilation.**  
Validated lifetime-acquired capabilities'ın functional genome'a aktarılması descendant zero-shot performance'ını artırır.

**H2 — Causal Gate.**  
Causal validation sonrası inheritance, her başarılı skill'i otomatik olarak inherit etmekten daha iyi generalization ve daha düşük regression üretir.

**H3 — Learning Cost Reduction.**  
Assimilated traits, descendant agentların aynı capability'yi yeniden keşfetmek için ihtiyaç duyduğu action/token/evaluation miktarını azaltır.

**H4 — Cross-Backbone Heritability.**  
Functional genome'un belirli bir kısmı farklı foundation modellere taşındığında performans avantajının bir bölümünü korur.

**H5 — Stability.**  
Controlled assimilation, unrestricted persistent skill accumulation'a göre daha düşük genome bloat ve daha az catastrophic behavioural regression üretir.

**H6 — Generational Improvement.**  
Population fitness'i belirli generation aralığında sistematik olarak başlangıç population'ının üzerine çıkar.

---

# 16. Ana baseline'lar

HeritAgent kendisiyle karşılaştırılmayacaktır.

Minimum experimental conditions:

```text
A. Static Agent
   No persistent learning.

B. Reflection Agent
   Learns/revises within session only.

C. Persistent Skill Agent
   Acquired skills are stored and directly reused.

D. Evolutionary Agent
   Mutation + selection + optional recombination,
   but no somatic/germline distinction and no CIG.

E. HeritAgent
   Somatic acquisition
   + causal inheritance gate
   + validated assimilation
   + evolutionary population.
```

Ablation studies ayrıca:

```text
HeritAgent – CIG
HeritAgent – generalization gate
HeritAgent – regulatory genes
HeritAgent – lifetime learning
HeritAgent – population diversity
```

şeklinde yapılmalıdır.

---

# 17. En kritik deney

Paper'ın en önemli karşılaştırması şudur:

```text
UNRESTRICTED INHERITANCE

experience
→ extracted skill
→ inherit
```

karşı:

```text
HERITAGENT

experience
→ candidate trait
→ replay
→ ablation
→ attribution
→ held-out validation
→ inherit
```

Bizim gerçek research contribution'ımız ancak ikinci sistemin:

```text
better held-out performance
lower regression
smaller / cleaner genome
lower descendant adaptation cost
```

ürettiğini gösterirsek güçlü hale gelecektir.

---

# 18. Yeni metrikler

HeritAgent'ın kendi metrics layer'ı olacaktır.

### Assimilation Gain

\[
AG =
F(D_{inherit})-
F(D_{no-inherit})
\]

Trait inheritance'ın descendants üzerindeki etkisi.

### Inheritance Precision

Genome'a alınan trait'lerin gerçekten positive contribution gösteren oranı.

### Genome Efficiency

\[
GE =
\frac{Performance}{GenomeComplexity}
\]

Daha büyük genome'un otomatik olarak daha iyi kabul edilmesini engeller.

### Descendant Adaptation Cost

Descendant'ın yeni görev dağılımına adapte olmak için kullandığı:

```text
actions
tokens
episodes
retries
evaluation compute
```

miktarı.

### Cross-Backbone Heritability

Bir model üzerinde evolve edilen genome'un başka foundation modelde koruduğu normalized gain.

### Regression Rate

Yeni assimilated trait sonrasında daha önce çözülmüş görevlerde bozulan performans.

---

# 19. Cross-backbone deneyi

Bu deney özellikle önemlidir.

Örneğin:

```text
Evolution backbone:
Qwen
```

üzerinde `Genome G20` oluşturulur.

Sonra aynı executable genome:

```text
Gemma + G20
Mistral + G20
```

olarak değerlendirilir.

Hipotetik sonuç:

```text
Gemma base      51%
Gemma + G20     58%
```

olursa functional genome'un foundation-model-independent bilgi taşıdığına dair önemli evidence oluşur.

Sonuç negatif çıkarsa da değerli olacaktır: genotype'ın model-specific expression sorununu gösterebilir.

Bu nedenle paper sonucu baştan varsayılmamalıdır.

---

# 20. Benchmark stratejisi

İlk HeritAgent paper'ı domain-specific olmamalıdır.

0rce veya cybersecurity başlangıç benchmark'ı olmayacaktır.

İlk deney paketinin en az üç farklı agent capability sınıfı içermesi tercih edilir:

```text
coding
tool/environment interaction
reasoning/planning
```

Amaç HeritAgent'ın yalnızca belirli bir task setinin optimizer'ı olmadığını göstermek olacaktır.

Security/Predator daha sonra HeritAgent'ın ayrı application paper'ı olabilir.

---

# 21. Foundation model politikası

İlk engine mümkün olduğunca model-independent tasarlanacaktır.

Model adapter interface:

```text
ModelAdapter
├── generate()
├── structured_generate()
├── tool_call()
├── estimate_tokens()
└── metadata()
```

Desteklenebilecek backends:

```text
local Qwen
Gemma
Mistral
OpenAI-compatible APIs
Anthropic-compatible APIs
future local models
```

HeritAgent araştırmasının kendisi belirli bir proprietary API'nin davranışına bağımlı olmamalıdır.

---

# 22. Sistem mimarisi

Ana architecture:

```text
                     ┌─────────────────┐
                     │ Foundation LLM  │
                     │     frozen      │
                     └────────┬────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Executable Agent    │
                   │ Genome — EAG        │
                   └──────────┬──────────┘
                              │ expression
                              ▼
                    ┌──────────────────┐
                    │ Runtime Agent    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Environment      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Trajectory       │
                    │ Recorder         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Trait Miner      │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Somatic Trait Store  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Causal Inheritance   │
                  │ Gate — CIG           │
                  │                      │
                  │ replay               │
                  │ ablation             │
                  │ attribution          │
                  │ generalization       │
                  │ regression           │
                  └──────────┬───────────┘
                             │
                      validated trait
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Germline Registry    │
                  └──────────┬───────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Evolution Engine       │
                 │                        │
                 │ selection              │
                 │ mutation               │
                 │ recombination          │
                 │ diversity              │
                 └───────────┬────────────┘
                             │
                             ▼
                       Generation t+1
```

---

# 23. Yazılım bileşenleri

Repo'nun önerilen yapısı:

```text
heritagent/
│
├── genome/
│   ├── schema/
│   ├── expression/
│   ├── regulation/
│   └── validation/
│
├── runtime/
│   ├── agent/
│   ├── model_adapters/
│   ├── tool_adapters/
│   └── context/
│
├── trajectory/
│   ├── recorder/
│   ├── normalization/
│   └── storage/
│
├── traits/
│   ├── miner/
│   ├── somatic_store/
│   ├── registry/
│   └── schemas/
│
├── inheritance/
│   ├── replay/
│   ├── ablation/
│   ├── attribution/
│   ├── generalization/
│   └── gate/
│
├── evolution/
│   ├── population/
│   ├── mutation/
│   ├── recombination/
│   ├── selection/
│   └── diversity/
│
├── lineage/
│   ├── graph/
│   └── provenance/
│
├── evaluation/
│   ├── fitness/
│   ├── metrics/
│   ├── benchmarks/
│   └── ablations/
│
├── experiments/
│   ├── manifests/
│   ├── runners/
│   └── results/
│
├── research/
│   ├── related-work/
│   ├── novelty-ledger/
│   ├── hypotheses/
│   └── paper/
│
└── tests/
```

---

# 24. Deneylerin reproducibility standardı

Her experiment immutable manifest ile tanımlanmalıdır.

```yaml
experiment:
  id: EXP-0142
  hypothesis: H2

model:
  name: qwen-example
  revision: sha...

genome:
  revision: G-0042

population:
  size: 16
  generation: 8

environment:
  benchmark: ...
  split: heldout-v2

inheritance:
  causal_gate: true
  ablation_replays: 5
  generalization_tasks: 20

seed:
  - 11
  - 23
  - 41
```

Sonuçlar seed, model revision, genome revision, dataset split ve code commit olmadan raporlanmamalıdır.

Bu paper için kritik olacaktır.

---

# 25. Paper'ın omurgası

Çalışma makaleye dönüştüğünde önerilen yapı:

## Abstract

Problem → existing gap → HeritAgent → method → experiments → gerçek sonuç rakamları.

Sonuç yokken rakam veya üstünlük iddiası yazılmayacaktır.

## 1. Introduction

Self-evolving agents, persistent skills ve architecture search gelişmiştir; fakat lifetime-acquired bir capability'nin hangi koşullarda kalıtsal agent architecture'a dönüşmesi gerektiği açık bir problem olarak konumlandırılacaktır.

## 2. Related Work

Agent architecture search, self-improving agents, evolutionary agent systems, skill learning, agent memory, contribution attribution, genotype/phenotype agent representations ve evolutionary computation ayrı alt alanlar olarak ele alınacaktır.

## 3. Problem Formulation

Genome, phenotype, environment, somatic trait, germline trait, fitness ve assimilation matematiksel olarak tanımlanacaktır.

## 4. HeritAgent Architecture

EAG, Trait Miner, Somatic Store, CIG, Germline Registry, Evolution Engine ve Lineage Graph açıklanacaktır.

## 5. Causal Inheritance

Replay, ablation, attribution, generalization ve regression testing metodolojisi ayrıntılandırılacaktır.

## 6. Evolutionary Process

Population, mutation, selection, recombination ve generation lifecycle verilecektir.

## 7. Experimental Setup

Models, benchmarks, baselines, seeds, compute budget ve statistical testing açıklanacaktır.

## 8. Main Results

H1–H6 sonuçları.

## 9. Ablations

CIG, lifetime learning, regulation, diversity vb. bileşenlerin katkıları.

## 10. Analysis

Genome evolution, lineage, trait survival, genome bloat, failed assimilations ve cross-backbone transfer.

## 11. Limitations

Compute cost, noisy causal attribution, benchmark overfitting, model dependence, trait interactions.

## 12. Conclusion

HeritAgent'ın hangi iddialarının deneysel olarak desteklendiği açık biçimde belirtilecektir.

---

# 26. Paper'ın Figure 1'i

Paper'ın ana görseli şu ilişkiyi anlatmalıdır:

```text
             INHERITED GENOME
                    │
                    ▼
                  AGENT
                    │
              interacts with
                    │
                    ▼
               ENVIRONMENT
                    │
                    ▼
               TRAJECTORY
                    │
                    ▼
            LIFETIME LEARNING
                    │
                    ▼
             ACQUIRED TRAIT
                    │
                    ▼
        CAUSAL INHERITANCE GATE
            /               \
        REJECT             ASSIMILATE
                             │
                             ▼
                    NEXT-GEN GENOME
                             │
                             ▼
                       DESCENDANTS
```

Figure tek bakışta paper'ın katkısını açıklamalıdır.

---

# 27. İlk prototype'ta neyi yapmayacağız?

İlk iteration'da gereksiz complexity eklenmeyecektir.

Multi-agent ecology, neural-weight evolution, RL training, custom foundation model training, 0rce integration, huge-scale open-ended evolution ve onlarca domain ilk milestone değildir.

İlk prototype şu soruyu cevaplamalıdır:

> Bir agent task sırasında yeni bir reusable trait öğrendiğinde, CIG bu trait'in kalıtıma değer olup olmadığını güvenilir biçimde ayırt edebiliyor mu ve bu seçim descendant performansına fayda sağlıyor mu?

Bu çalışmıyorsa population genetics'in geri kalanını büyütmenin anlamı yoktur.

---

# 28. Geliştirme sırası

Projenin mühendislik geliştirmesi aşağıdaki sequence'i izlemelidir:

```text
EAG schema
↓
single-agent runtime
↓
trajectory recorder
↓
trait miner
↓
somatic trait store
↓
replay evaluator
↓
ablation evaluator
↓
basic CIG
↓
single-lineage inheritance
↓
generation comparison
↓
population manager
↓
mutation
↓
selection
↓
recombination
↓
full lineage graph
↓
cross-backbone experiments
```

Bu sıralama önemlidir.

Evolution engine'i ilk günden yapmak yerine **inheritance mechanism önce doğrulanmalıdır.**

---

# 29. İlk milestone

**Milestone 0 — Minimal HeritAgent**

Tek model.

Tek domain.

Tek agent.

İki generation.

Akış:

```text
G0 agent
↓
task set
↓
learn candidate trait
↓
replay
↓
ablation
↓
CIG
↓
trait assimilated
↓
G1 agent
↓
held-out task set
```

Minimum başarı kriteri:

G1'in held-out performance'ında reproducible pozitif improvement göstermesi ve unrestricted-skill baseline'a göre daha düşük regression göstermesidir.

Bu başarıldıktan sonra population-based evolution'a geçilir.

---

# 30. Research discipline

Bu proje hızlı hareket eden bir araştırma alanındadır.

2026 içinde bile SkillRL, Mendel Gödel Machine, SkillShapley ve benzeri çalışmalar çıkmıştır. Bu nedenle novelty bir defa araştırılıp bırakılmamalıdır.

Repo içinde yaşayan bir:

```text
research/novelty-ledger.md
```

dosyası tutulacaktır.

Her ilgili paper için:

```text
paper
date
core mechanism
overlap with HeritAgent
what remains different
required architecture changes
```

kaydedilecektir.

Yeni paper çıktığında novelty claim gerekiyorsa değiştirilmelidir.

Bilimsel hedef fikri “korumak” değil, gerçekten açık problemi çözmektir.

---

# 31. İki-agent ortak çalışma protokolü

Bu proje iki AI agent ve insan araştırmacının ortak geliştirmesiyle yürütülecektir.

Chat geçmişleri **source of truth değildir.**

Source of truth repo'dur.

Her önemli karar aşağıdakilerden birine yazılmalıdır:

```text
research/decisions/
research/novelty-ledger.md
research/hypotheses/
experiments/manifests/
architecture/
```

Agentlar task başlatmadan önce ilgili dosyaları okumalıdır.

Bir agent architecture değişikliği önerdiğinde yalnızca kod değiştirmemeli; değişiklik şu formatta kayıt altına alınmalıdır:

```text
Problem
Current design
Proposed change
Reason
Research implication
Experiment needed
Compatibility impact
```

Araştırma iddiası yalnızca benchmark sonucuyla desteklenebilir.

“Daha iyi görünüyor”, “muhtemelen çalışır”, “literatürde yok gibi” ifadeleri paper claim'ine dönüşmemelidir.

---

# 32. Agent görev ayrımı

İki agent katı biçimde birbirinden izole edilmeyecektir; ancak doğal bir rol ayrımı faydalıdır.

**Research / Critic role** ağırlıklı çalışan agent:

```text
literature monitoring
novelty analysis
hypothesis design
experimental validity
statistics
paper structure
claim auditing
```

konularına odaklanabilir.

**Systems / Builder role** ağırlıklı çalışan agent:

```text
architecture
schemas
runtime
CIG implementation
population engine
benchmark integration
experiment automation
profiling
```

konularına odaklanabilir.

Fakat kritik kararlar cross-review edilmelidir.

Research agent engineering design'i eleştirebilmeli; builder agent research hypothesis'in uygulanabilirliğini sorgulayabilmelidir.

Hiçbir agent kendi ürettiği kritik sonucu tek başına “validated” ilan etmemelidir.

---

# 33. Deney review protokolü

Her major experiment için iki ayrı aşama olacaktır.

**Pre-registration style review**

Experiment çalıştırılmadan önce:

```text
hypothesis
dependent variable
independent variable
baseline
dataset split
seeds
success/failure condition
```

kilitlenir.

**Post-run review**

Sonuç geldikten sonra:

```text
expected result
actual result
statistical significance
unexpected behaviour
confounders
follow-up
```

kaydedilir.

Bu özellikle confirmation bias'ı azaltacaktır.

---

# 34. Anti-overclaim kuralları

Aşağıdaki ifadeler experimental evidence olmadan kullanılmamalıdır:

```text
first
novel
better
causal
general
model-independent
evolutionary improvement
inheritance
```

Özellikle “causal” kelimesi yalnızca controlled interventions/ablations gerçekten yapılırsa kullanılacaktır.

Aksi durumda “contribution estimate” veya “association” denmelidir.

---

# 35. 0rce ile ilişkisi

HeritAgent başlangıçta 0rce'den bağımsız, domain-general bir research engine olarak geliştirilecektir.

Bu stratejik olarak önemlidir.

Daha sonra:

```text
HeritAgent
      │
      ▼
security environment
      │
      ▼
0rce Predator
```

şeklinde uygulanabilir.

Predator'ın farklı generations boyunca:

```text
planning strategies
recon policies
evidence processing
tool policies
failure recovery
attack-state handling
```

gibi functional traits edinmesi ve yalnızca doğrulanmış capability'leri gelecek generations'a aktarması mümkün olabilir.

Bu, ayrı bir application/system paper için güçlü temel oluşturabilir.

Ancak HeritAgent'ın ilk paper'ı Predator'a bağımlı olmayacaktır.

---

# 36. Uzun vadeli research program

Araştırma hattı doğal olarak üç aşamaya ayrılabilir:

```text
HERITAGENT
Foundational mechanism:
acquisition → causal validation → inheritance

        ↓

HERITAGENT POPULATIONS
Open-ended / population-level
multi-generational agent evolution

        ↓

0RCE PREDATOR
Domain-specific application of
heritable functional agent evolution
```

Böylece 0rce için geliştirilen uygulama temel araştırmanın test alanlarından biri olur; temel teori yalnızca security bağlamına sıkışmaz.

---

# 37. Projenin başarı tanımı

HeritAgent'ın başarısı:

> “Agent kendi kendini değiştirdi.”

değildir.

Başarılı sayılabilmesi için deneysel olarak şu zincirin gösterilmesi gerekir:

```text
1. Agent lifetime içinde yeni capability kazanıyor.

2. Sistem capability'yi tekrar kullanılabilir trait olarak çıkarıyor.

3. Controlled experiments trait'in faydasını ölçüyor.

4. Faydalı trait germline genome'a assimilate ediliyor.

5. Descendants trait'i tekrar öğrenmeden kullanabiliyor.

6. Descendants held-out görevlerde measurable advantage gösteriyor.

7. Selective inheritance,
   indiscriminate skill accumulation'dan daha iyi sonuç veriyor.

8. Süreç birden fazla generation boyunca tekrarlanabiliyor.
```

Bu sekiz adım gösterildiğinde HeritAgent'ın asıl research claim'i güçlü biçimde desteklenmiş olur.

---

# 38. Çalışmanın öz cümlesi

Projede yön kaybedildiğinde şu cümleye dönülmelidir:

> **HeritAgent investigates whether capabilities learned by an LLM agent during its lifetime can be causally validated, assimilated into an explicit executable genome, and inherited by future generations to produce measurable cross-generational improvement without modifying the underlying foundation model.**

Her mimari karar bu soruya hizmet etmelidir.

---

# 39. Şu anki çalışma başlığı

**HeritAgent: Causally-Gated Assimilation of Acquired Skills into Functional Genomes for Evolvable LLM Agents**

Bu başlık working title'dır.

Çalışmanın sonuçları ve ilgili literatürdeki gelişmelere göre submission öncesinde değiştirilebilir.

---

# 40. İlk görev

Projeyi alan yeni agentın ilk işi kod yazmak değildir.

Önce:

```text
1. Bu specification'ı oku.
2. AgentSquare'i incele.
3. Darwin Gödel Machine'i incele.
4. Mendel Gödel Machine'i incele.
5. Genomebook'u incele.
6. SkillRL'i incele.
7. ShapleyFlow/CapaBench'i incele.
8. SkillShapley'i incele.
9. novelty-ledger oluştur/güncelle.
10. EAG v0 schema öner.
11. Minimal CIG experiment'ını tasarla.
12. Ancak bundan sonra implementation başlat.
```

Mevcut literatürün modüler agent evolution, lineage/self-modification, controlled inheritance, behavioural genetic representation, recursive skill evolution ve contribution attribution konularında halihazırda güçlü mekanizmalar sunduğu unutulmamalıdır. HeritAgent bunları yeniden icat etmeye değil, **lifetime-acquired functional capabilities için kontrollü ve ölçülebilir bir inheritance mechanism eksikliğini** araştırmaya odaklanmalıdır.