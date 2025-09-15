# XGBoost Production Deployment Plan

## Summary

**Coverage**: 18/21 symbols ready for production (>1.0 Sharpe)
**Asset Classes**: Full coverage across EQUITY, RATES, FX, METALS, AGS

## Production-Ready Symbols

### Tier 1: Excellent (>2.0 Sharpe) - 8 symbols
```bash
QGC#C: 2.837    # Gold
@NQ#C: 2.385    # NASDAQ
@ES#C: 2.319    # S&P500
BL#C: 2.290     # Euro Bund
@RTY#C: 2.193   # Russell
@TY#C: 2.067    # US 10Y
BD#C: 2.050     # Euro Bund
QPL#C: 2.047    # Platinum
```

### Tier 2: Good (1.5-2.0 Sharpe) - 7 symbols
```bash
@AD#C: 1.990    # Australian Dollar
@S#C: 1.985     # Soybeans
@KW#C: 1.981    # Wheat
@W#C: 1.837     # Wheat
@US#C: 1.837    # US 30Y
@EU#C: 1.769    # Euro
@FV#C: 1.721    # US 5Y
```

### Tier 3: Acceptable (1.0-1.5 Sharpe) - 3 symbols
```bash
QSI#C: 1.592    # Silver
@SM#C: 1.456    # Soy Meal
QHG#C: 1.320    # Copper
```

## Deployment Strategy

### Phase 1: Core Portfolio (Tier 1 only)
**8 symbols with >2.0 Sharpe for initial deployment**

### Phase 2: Full Portfolio (Tiers 1+2)
**15 symbols with >1.5 Sharpe for complete deployment**

### Phase 3: Extended Portfolio (All tiers)
**18 symbols for maximum diversification**

## Model Building Commands

### Tier 1 (Priority)
```bash
# Build core excellent performers first
python production_model_builder.py --symbol "QGC#C"
python production_model_builder.py --symbol "@NQ#C"
python production_model_builder.py --symbol "@ES#C"
python production_model_builder.py --symbol "BL#C"
python production_model_builder.py --symbol "@RTY#C"
python production_model_builder.py --symbol "@TY#C"
python production_model_builder.py --symbol "BD#C"
python production_model_builder.py --symbol "QPL#C"
```

### Tier 2 (Secondary)
```bash
python production_model_builder.py --symbol "@AD#C"
python production_model_builder.py --symbol "@S#C"
python production_model_builder.py --symbol "@KW#C"
python production_model_builder.py --symbol "@W#C"
python production_model_builder.py --symbol "@US#C"
python production_model_builder.py --symbol "@EU#C"
python production_model_builder.py --symbol "@FV#C"
```

### Tier 3 (Optional)
```bash
python production_model_builder.py --symbol "QSI#C"
python production_model_builder.py --symbol "@SM#C"
python production_model_builder.py --symbol "QHG#C"
```

## Production Configuration

### Update trading_config.yaml
- Enable only Tier 1+2 symbols initially (15 symbols)
- Set max_traded > 0 for active symbols
- Set max_traded = 0 for excluded symbols (@BO#C, @C#C, @CT#C)

### Folder Structure
```
PROD/
├── models/              # Top 15 models per symbol
│   ├── @ES#C/          # 15 × 0.5MB = ~8MB
│   ├── @TY#C/          # 15 × 0.5MB = ~8MB
│   └── ...             # Total: ~120MB vs 9GB
├── config/models/       # Clean YAML configs
└── ...
```

## Expected Storage
- **Per Symbol**: ~8MB (15 models vs 600MB for 150 models)
- **Total (18 symbols)**: ~150MB vs 10GB+
- **Reduction**: 98% storage savings

## Next Steps
1. **Build Tier 1 models** (8 symbols - core portfolio)
2. **Test with synthetic data**
3. **Deploy Tier 1 to production**
4. **Gradually add Tier 2 symbols**
5. **Monitor performance vs benchmarks**

---
*Generated: 2025-09-15*
*Strategy: Phased deployment starting with highest Sharpe performers*