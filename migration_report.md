# Aadhar Migration Prediction Report
**Generated:** 2026-01-09 19:31:23
---
## Executive Summary
Based on current news activity and historical Aadhar demographic update patterns, **Karnataka** has the highest predicted migration probability at **16.95%**.

---
## 1. Data Overview
### Aadhar Demographic Data
- **Total Records:** 735,747
- **States Analyzed:** 6
- **Date Range:** 2025-03-01 00:00:00 to 2026-01-03 00:00:00

### News Data
- **Total Articles Fetched:** 2,793
- **Search Keywords:** jobs, migration, hiring, industrial growth, layoffs

---
## 2. State-wise Data Summary
| State | Months | Avg Monthly Updates | Total News Articles |
|-------|--------|---------------------|---------------------|
| Andhra Pradesh | 10 | 228,415 | 340 |
| Delhi | 10 | 130,719 | 272 |
| Haryana | 10 | 125,252 | 271 |
| Karnataka | 10 | 161,840 | 324 |
| Maharashtra | 10 | 494,527 | 333 |

---
## 3. Prediction Model Analysis
### Model Details
- **Algorithm:** Custom Linear Regression (NumPy-based)
- **Training Samples:** 40
- **Test Samples:** 10
- **Root Mean Square Error:** 652,226

### Key Factors Influencing Migration
| Factor | Impact | Interpretation |
|--------|--------|----------------|
| Hiring News | STRONG + | High migration when more news |
| Industrial Growth News | STRONG + | High migration when more news |
| Jobs News | STRONG - | Lower migration when more news |
| Layoffs News | STRONG - | Lower migration when more news |
| Migration News | Weak + | Slight increase with more news |

**Strongest Correlation:** Hiring News (22.95% correlation)

### Reasoning Behind the Model
The model uses news activity as a leading indicator of migration patterns. The logic is:
- **Hiring/Industrial Growth News** → Economic opportunities → Attracts migrants
- **Jobs News** → May indicate competition/saturation → Mixed effect
- **Layoffs News** → Economic distress → Discourages migration or triggers outflux
- **Migration News** → Already happening events → Lagging indicator

---
## 4. Migration Prediction Results
### State Rankings by Migration Probability
| Rank | State | Probability | Analysis |
|------|-------|-------------|----------|
| 1 | Karnataka | 16.95% | Low migration activity expected |
| 2 | Andhra Pradesh | 16.94% | Low migration activity expected |
| 3 | Telangana | 16.93% | Low migration activity expected |
| 4 | Maharashtra | 16.92% | Low migration activity expected |
| 5 | Delhi | 16.76% | Low migration activity expected |
| 6 | Haryana | 15.51% | Low migration activity expected |

---
## 5. Conclusion & Recommendations
### Top Migration Hotspots
1. **Karnataka** (16.95%)
2. **Andhra Pradesh** (16.94%)
3. **Telangana** (16.93%)

### Recommendations
- Monitor Aadhar update centers in high-probability states for increased activity
- Allocate resources proportionally to predicted migration probabilities
- Track news trends weekly to update predictions

---
