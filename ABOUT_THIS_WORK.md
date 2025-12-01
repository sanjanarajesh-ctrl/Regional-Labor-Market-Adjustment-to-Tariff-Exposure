# Trade Policy Shocks and Regional Resilience: Evidence from the US–China Tariffs

This repository contains the code, data pipeline, and empirical results for a project studying how United States counties responded to the 2018–2019 US–China tariff war.

The core question is:

> When trade policy suddenly raises tariffs on key sectors, which local economies absorb the shock, which ones lag, and what county characteristics predict resilience?

Using a county–year panel from 2015 to 2019, I construct a tariff exposure index and trace its impact on employment, GDP, and county level “resilience” measures. The work combines tools from the trade shock literature with the regional resilience literature to understand how policy driven trade shocks play out across space.

## 1. Motivation

The 2018–2019 US–China tariffs were large, sudden, and highly visible. We now have very good evidence on:

- aggregate incidence and welfare losses,  
- sectoral impacts on manufacturing and exporters,  
- some regional effects on consumption and coarse labor market outcomes.

What we do **not** have is a systematic, county level picture of:

1. how tariff exposure translated into employment and GDP changes, and  
2. how those effects varied with pre shock county characteristics like income, education, and population mobility.

This project fills that gap. It uses the tariff war as a natural experiment in “regional resilience” to trade policy shocks, asking which counties manage to adapt and which do not.

## 2. Data and construction

The analysis is based on a county by year panel for 2015–2019. The main dataset is:

- `data/processed/trade_shock_panel_county_2015_2019.csv`

Each county–year observation includes:

- **Tariff exposure index**  
  - USTR Section 301 tariff lists mapped from HTS to NAICS.  
  - NAICS sectors mapped to counties using BLS QCEW employment shares.  
  - Exposure is a continuous index of how intensively a county’s employment is concentrated in tariff targeted sectors and their downstream users.

- **Outcomes**  
  - `log_emp`: log total employment (BLS QCEW).  
  - `log_gdp`: log county GDP (BEA).  
  - `delta_log_emp_17_19`: log employment change between 2017 and 2019.

- **Firm dynamics**  
  - Business Dynamics Statistics (Census BDS): establishment entry, exit, and employment dynamics.

- **Demographics and controls** (ACS)  
  - Median household income.  
  - Share of adults with BA or higher.  
  - Share of residents who moved from another state (interstate mobility).  

The panel is built by `code/build_panel.py`, which:

1. Ingests raw data from USTR, QCEW, BDS, BEA, and ACS.  
2. Harmonizes industry codes.  
3. Constructs the tariff exposure index at the county level.  
4. Merges all sources into a single panel and writes  
   `data/processed/trade_shock_panel_county_2015_2019.csv`.

## 3. Empirical strategy

The empirical design follows the trade shock literature but adapts it to a policy driven tariff increase.

### 3.1 Baseline difference in differences

Baseline specification:

\[
\log Y_{ct} = \alpha_c + \lambda_t + \beta \left(\text{Exposure}_c \times \text{Post}_{t}\right) + X_{ct}'\gamma + \varepsilon_{ct}
\]

- \(Y_{ct}\) is log employment or log GDP in county \(c\) and year \(t\).  
- \(\alpha_c\) are county fixed effects.  
- \(\lambda_t\) are year fixed effects.  
- \(\text{Exposure}_c\) is the tariff exposure index.  
- \(\text{Post}_t\) equals 1 in 2018–2019 and 0 in 2015–2017.  
- Standard errors are clustered by county.

This delivers a difference in differences estimate of how more exposed counties diverged from less exposed counties after the tariffs.

### 3.2 Event study

To examine timing and pre trends, I estimate:

\[
\log Y_{ct} = \alpha_c + \lambda_t + 
\sum_{k \neq 0} \beta_k \left(\text{Exposure}_c \times 1\{\text{Year}=2017+k\}\right) + X_{ct}'\gamma + \varepsilon_{ct}
\]

which yields event time coefficients \(\beta_k\) for years \(k\) relative to 2017.  
These are plotted as Figure 1 (employment) and Figure 2 (GDP).

### 3.3 Heterogeneity and resilience

To study resilience, I interact exposure with pre shock county characteristics:

- High income vs low income counties.  
- High BA share vs low BA share.  
- High interstate mobility vs low mobility.

I also estimate a cross sectional “resilience regression”:

\[
\Delta \log(\text{Emp}_{c,17\to19}) = \theta_0 
+ \theta_1 \text{Exposure}_c 
+ \theta_2 \text{Exposure}_c \times \text{HighIncome}_c
+ \theta_3 \text{Exposure}_c \times \text{HighBA}_c
+ \theta_4 \text{Exposure}_c \times \text{HighMobility}_c
+ u_c
\]

where the dependent variable is log employment growth from 2017 to 2019.

## 4. Main results

The headline findings are:

1. **No large average employment collapse.**  
   Baseline difference in differences estimates show that, on average, more exposed counties did not experience a statistically significant employment decline relative to less exposed counties once county and year fixed effects are included. Point estimates are modest, and confidence intervals are wide enough to rule out very large negative effects.

2. **GDP differences pre date the tariffs.**  
   For GDP, exposed counties start the sample period below less exposed counties, and those differences show up in the pre period event study coefficients. Once pre trends are accounted for, there is no sharp additional GDP collapse after the tariffs, which suggests that the trade war interacted with an already uneven landscape rather than creating a new divide.

3. **Resilience is highly uneven.**  
   The most robust signal is heterogeneity:

   - In high income counties, greater tariff exposure is associated with **positive** employment responses.  
   - In high BA counties, exposure is also associated with **relative employment gains**.  
   - In low income and low BA counties, exposure has small and imprecise effects.

   In other words, counties that were already richer and more educated appear to convert exposure into opportunities, possibly through supply chain reconfiguration or attraction of new activity, while poorer and less educated counties do not.

4. **Cross sectional resilience regression confirms the pattern.**  
   When I regress 2017–2019 employment growth directly on exposure and its interactions, the main exposure term is close to zero, but the interaction with high income is positive and borderline significant. The interaction with high BA is positive but less precise, and the mobility interaction is small. This matches the panel evidence: resilience is most strongly associated with income and human capital.

Taken together, these results suggest that the 2018–2019 tariffs did not generate a China shock style employment collapse, but they did create a pattern where counties with stronger pre shock endowments were better able to adapt and even benefit at the margin.

## 5. Interpretation and contribution

This project brings together four components:

- Tariff incidence and the US–China trade war, which has focused on aggregate welfare and sectoral outcomes.  
- The local labor market literature on long run trade shocks, which has documented large and persistent regional effects of import competition and liberalization.  
- Trade policy uncertainty and firm level adaptation, which emphasizes how expectations and adjustment costs shape responses to policy.  
- The regional economic resilience literature, which links pre shock income, education, and mobility to the capacity to absorb shocks.

The contribution is to use the 2018–2019 tariffs as a laboratory for regional resilience at the county level. The results show that:

- Average employment effects of tariff exposure are small,  
- GDP differences reflect a mix of pre existing disadvantage and modest additional changes, and  
- The key margins of resilience are income and human capital rather than trade exposure alone.

From a policy perspective, this suggests that trade policy alone is not a reliable tool for reviving “left behind” counties. In this episode, tariff exposure did not systematically help poorer or less educated places catch up. Instead, it tended to reinforce existing strengths in counties that were already relatively advantaged.


