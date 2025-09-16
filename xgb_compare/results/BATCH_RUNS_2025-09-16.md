# Batch Runs - 2025-09-16

## Configuration
- **Models**: 100 per fold
- **Folds**: 15
- **Features**: 100 (limited)
- **Signal Type**: Binary (+1/-1)
- **XGB Type**: Standard
- **Q-Metric**: Sharpe
- **Exports**: Model (.pkl) + Signal Distribution (.csv)

## Results Matrix

| Symbol | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Test Status |
|--------|-----------------|--------------|-------------------|----------------|----------------------|-------------------|-------------|
| @AD#C  | 2.126          | 54.8%        | 1.983             | 54.3%          | 2.043                | 54.6%             | ✅ 233143   |
| @BO#C  | -0.148         | 46.2%        | 1.047             | 53.0%          | -0.087               | 49.0%             | ✅ 233203   |
| @BP#C  | 1.487          | 54.1%        | 1.318             | 51.2%          | 1.410                | 52.9%             | ✅ 233142   |
| @C#C   | 1.440          | 51.9%        | 1.112             | 51.7%          | 1.313                | 51.8%             | ✅ 233330   |
| @CT#C  | 1.191          | 52.1%        | 0.577             | 51.7%          | 0.954                | 51.9%             | ✅ 233330   |
| @ES#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @EU#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @FV#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @JY#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @KW#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @NQ#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @RTY#C | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @S#C   | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @SM#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @TY#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @US#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| @W#C   | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| BD#C   | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| BL#C   | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| QBZ#C  | -              | -            | -                 | -              | -                    | -                 | ⚡ Running   |
| QCL#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QGC#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QHG#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QNG#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QPL#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QRB#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |
| QSI#C  | -              | -            | -                 | -              | -                    | -                 | 🔄 Queued    |

## Status Legend
- ⚡ Running: Test in progress
- 🔄 Queued: Waiting to start
- ✅ HHMMSS: Completed at timestamp
- ❌ Failed: Test encountered error

---
*Created: 2025-09-16 23:50*
*Last updated: 2025-09-16 23:50*