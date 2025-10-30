# DataLens: AI-Powered Data Analysis Platform
## Technical Overview & Architecture

---

## System Overview

```mermaid
graph TD
    U[User] --> |Upload Data| W[Web Interface]
    W --> |Process Data| DA[Data Analysis Engine]
    DA --> |Query| VAI[Vertex AI/PaLM]
    DA --> |Store| DD[Data & Analysis]
    DA --> |Generate| V[Visualizations]
    W --> |Display| V
    W --> |Show Results| U
    
    subgraph Authentication
        WIF[Workload Identity Federation]
        GC[Google Cloud]
        WIF --> |Authenticate| GC
    end
    
    DA --> |Authenticate| WIF
```

---

## Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant App
    participant WIF as Workload Identity Federation
    participant GCP as Google Cloud Platform
    participant VertexAI as Vertex AI

    User->>App: Start Application
    App->>WIF: Request Authentication
    WIF->>GCP: Validate Identity
    GCP->>WIF: Issue Token
    WIF->>App: Return Credentials
    App->>VertexAI: Initialize with Credentials
    VertexAI->>App: Confirmation
    App->>User: Ready for Analysis
```

---

## Data Analysis Process

```mermaid
graph LR
    subgraph Input
        U[Upload CSV] --> V[Validate Data]
    end
    
    subgraph Analysis
        V --> B[Basic Statistics]
        V --> P[Pattern Analysis]
        V --> O[Outlier Detection]
        V --> R[Relationships]
    end
    
    subgraph AI Processing
        B --> AI[PaLM Model]
        P --> AI
        O --> AI
        R --> AI
        AI --> I[Generate Insights]
    end
    
    subgraph Output
        I --> VIZ[Visualizations]
        I --> S[Summary]
        I --> BL[Business Logic]
    end
```

---

## Component Architecture

```mermaid
classDiagram
    class DataAnalysisAgent {
        +load_data()
        +generate_basic_stats()
        +analyze_patterns()
        +generate_business_logic()
        +answer_question()
        +generate_visualizations()
    }
    
    class WebInterface {
        +upload_data()
        +show_analysis()
        +show_patterns()
        +show_business_logic()
        +show_qa_interface()
    }
    
    class VertexAI {
        +initialize()
        +predict()
        +generate_text()
    }
    
    DataAnalysisAgent --> VertexAI
    WebInterface --> DataAnalysisAgent
```

---

## Business Logic Generation Flow

```mermaid
graph TD
    F[Field Selection] --> T[Type Analysis]
    T --> N[Numeric Analysis]
    T --> C[Categorical Analysis]
    T --> D[Date Analysis]
    
    N --> R[Generate Rules]
    C --> R
    D --> R
    
    R --> V[Validation Rules]
    R --> P[Pattern Rules]
    R --> CO[Constraint Rules]
    
    V --> PBL[Plain Business Logic]
    P --> PBL
    CO --> PBL
```

---

## User Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Interface
    participant Agent
    participant AI
    
    User->>Interface: Upload Data
    Interface->>Agent: Process Data
    Agent->>Agent: Generate Statistics
    Agent->>AI: Request Analysis
    AI->>Agent: Return Insights
    Agent->>Interface: Display Results
    
    loop Interactive Analysis
        User->>Interface: Ask Question
        Interface->>Agent: Process Query
        Agent->>AI: Generate Response
        AI->>Agent: Return Answer
        Agent->>Interface: Show Response
        Interface->>User: Display Answer
    end
```

---

## Data Processing Pipeline

```mermaid
graph LR
    subgraph Input Processing
        I[Input Data] --> V[Validation]
        V --> C[Cleaning]
    end
    
    subgraph Analysis Pipeline
        C --> S[Statistics]
        C --> P[Patterns]
        C --> O[Outliers]
    end
    
    subgraph AI Processing
        S --> AI[PaLM Model]
        P --> AI
        O --> AI
    end
    
    subgraph Output Generation
        AI --> VIZ[Visualizations]
        AI --> IN[Insights]
        AI --> BL[Business Logic]
        AI --> AN[Answers]
    end
```

---

## Security Architecture

```mermaid
graph TD
    subgraph Client
        UI[Web Interface]
    end
    
    subgraph Authentication
        WIF[Workload Identity Federation]
        UI --> |1. Request| WIF
        WIF --> |2. Validate| IDP[Identity Provider]
        IDP --> |3. Token| WIF
    end
    
    subgraph Google Cloud
        WIF --> |4. Auth| GCP[Google Cloud Platform]
        GCP --> |5. Access| VAI[Vertex AI]
        GCP --> |6. Access| DS[Data Services]
    end
```

---

## Features and Capabilities

```mermaid
mindmap
    root((DataLens))
        Data Upload
            CSV Support
            Validation
            Preview
        Analysis
            Basic Stats
            Patterns
            Outliers
            Relationships
        AI Features
            Q&A
            Insights
            Recommendations
        Business Logic
            Field Rules
            Validations
            Constraints
        Visualization
            Charts
            Graphs
            Heatmaps
```

---

## Development Timeline

```mermaid
gantt
    title Development Phases
    dateFormat  YYYY-MM-DD
    section Setup
    Environment Setup      :2025-11-01, 3d
    Web Interface         :2025-11-04, 4d
    section Core
    Data Analysis         :2025-11-08, 5d
    AI Integration        :2025-11-13, 4d
    section Features
    Business Logic        :2025-11-17, 4d
    Visualization         :2025-11-21, 3d
    section Final
    Testing              :2025-11-24, 5d
    Documentation        :2025-11-29, 3d
```

---

## Key Benefits

1. **Automated Analysis**
   - Quick insights from data
   - Consistent analysis patterns
   - Reduced manual effort

2. **AI-Powered Insights**
   - Natural language interaction
   - Context-aware responses
   - Intelligent pattern detection

3. **Business Logic Generation**
   - Automated rule creation
   - Field-specific validation
   - Clear documentation

4. **Security**
   - Workload Identity Federation
   - No stored credentials
   - Secure cloud integration

5. **Scalability**
   - Cloud-native architecture
   - Handles large datasets
   - Extensible framework

---

## Future Roadmap

```mermaid
graph TD
    C[Current Version] --> V1[Version 1.1]
    V1 --> V2[Version 1.2]
    V2 --> V3[Version 2.0]
    
    subgraph "Version 1.1"
        F1[Additional File Formats]
        F2[Enhanced Visualizations]
    end
    
    subgraph "Version 1.2"
        F3[Advanced AI Models]
        F4[Custom Rule Templates]
    end
    
    subgraph "Version 2.0"
        F5[Real-time Analysis]
        F6[Collaborative Features]
        F7[API Integration]
    end
```