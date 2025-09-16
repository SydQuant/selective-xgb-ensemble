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
| @ES#C  | 0.762          | 51.5%        | 1.370             | 53.5%          | 1.028                | 52.4%             | ✅ 003628   |
| @EU#C  | 2.051          | 54.6%        | 1.558             | 55.3%          | 1.778                | 54.9%             | ✅ 003124   |
| @FV#C  | 0.989          | 53.2%        | 1.869             | 55.1%          | 1.421                | 54.0%             | ✅ 002501   |
| @JY#C  | 0.395          | 49.5%        | 1.935             | 55.7%          | 1.205                | 52.1%             | ✅ 003627   |
| @KW#C  | 0.631          | 50.0%        | 1.571             | 51.4%          | 1.093                | 50.6%             | ✅ 003833   |
| @NQ#C  | 1.545          | 54.5%        | 1.824             | 54.1%          | 1.667                | 54.3%             | ✅ 003832   |
| @RTY#C | 0.430          | 51.0%        | 0.473             | 53.3%          | 0.448                | 51.9%             | ✅ 003833   |
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