# Литобзор: forward-PINN для density-wave неустойчивости в сверхкритике без supervised-данных

**Дата:** 2026-06-13
**Запрос:** работы, где density-wave неустойчивость (DWO) в сверхкритике ловится forward-моделью (PINN) без supervised-данных.

## Главный вывод

Точного попадания «forward PINN ловит DWO в сверхкритике без данных» в литературе **нет** — это зазор/новизна данной работы. Но есть 4 работы, закрывающие ровно те механизмы, на которых стоит задача (причинность обучения, достижимость неустойчивой ветки решения, нормировка невязки), плюс соседи по физике и архитектуре.

---

## Прямое ядро — методы, которые уже используются / должны цитироваться

| Работа | Claim | Почему критично |
|---|---|---|
| **Wang, Sankaran, Perdikaris — «Respecting causality for training PINNs»** (CMAME 2024) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000690) · [код](https://github.com/PredictiveIntelligenceLab/CausalPINNs) | Vanilla-PINN не воспроизводит multi-scale/хаос/turbulent, потому что NTK-bias заставляет минимизировать невязку сначала на ПОЗДНИХ временах → нарушает причинность. Causal weighting `w_m = exp(−ε·Σ_{j<m} L_j)` чинит. Первый раз PINN взял Lorenz, Kuramoto–Sivashinsky (хаос), Navier–Stokes. | Это ровно `causal_pde_loss`. Главный first-principles аргумент, почему гладкий forward-PINN садится на неустойчивое QS-решение и как это пробить. **Цитировать как фундамент.** |
| **Dong, Cao, Suo, Kou, Zhang — «Optimization-Based Discovery of a Non-Attracting Flow State in an Oscillating-Cylinder Wake»** (arXiv:2604.00441) | Forward-PINN на NS **без данных** находит решения, *недостижимые time-stepping'ом* — в т.ч. неустойчивый предельный цикл / вихревую дорожку Кармана при сверхкритическом Re. Есть приложение с канонической моделью Хопфа. Ключ: исход зависит от **scale инициализации** — при малой сеть застревает у нуля, не доходя до цикла. | Самая близкая по духу: PINN-как-forward-solver достаёт *non-attracting* неустойчивое решение без supervised. «Сверхкритический Re», предельный цикл, Хопф — прямая аналогия DWO. Предупреждение про инициализацию = warm-start/гейт. **Главный paper-аналог.** |

---

## Гипотезы про то, *почему forward-PINN не осциллирует* — и как чинят другие

| Работа | Гипотеза / механизм | Применимо |
|---|---|---|
| **«Resolving Sharp Gradients of Unstable Singularities to Machine Precision via Neural Networks»** (arXiv:2511.22819) | Стандартный PINN-loss минимизирует *абсолютную* невязку → размазывает ошибку равномерно, давит резкие/тонкие особенности, **застревает на высоком residual** у неустойчивых решений. Лечение: **gradient-normalized residual**. | Прямо проблема нормировки невязки по зонам (`energy_scale`, per-zone). Гипотеза: keepers — частный случай ребаланса landscape'а. Проверить gradient-norm как альтернативу ручным весам. |
| **«Predictive Limitations of PINNs in Vortex Shedding»** (arXiv:2306.00230) | Документирует, что vanilla-PINN **не предсказывает** периодический срыв/осцилляции — садится на стационар. | Контр-пример/мотивация: подтверждает, что без спец-механизма forward-PINN осцилляции теряет. Для раздела «почему наивный подход не работает». |
| **«Learning thermoacoustic interactions in combustors using a PINN»** (ScienceDirect 2024) — [link](https://www.sciencedirect.com/science/article/abs/pii/S095219762401546X) | PINN на low-order модели для **самовозбуждающихся** термоакустических колебаний (тоже Хопф-неустойчивость, предельный цикл). | Ближайший физический кузен DWO — другая, но та же математика (self-excited limit cycle). Источник идей по архитектуре/форсингу. NB: у них есть data-loss — контраст «а мы без данных». |

---

## Нейросети любого типа для DWO — прямые соседи целевой статьи (10.1016/j.net.2024.103407)

Целевой DOI — это **Buchanan et al., «A recurrent neural network for modeling natural circulation density wave instabilities»** (*Nuclear Engineering and Technology*, 2024) — LSTM/RNN, обученный на данных DWO. Ниже — кластер работ, где DWO ловят нейросетью **любого типа**. Важно: **все они supervised / data-driven** (учатся на данных кода или эксперимента); ни одна не forward-physics. Это прямой контраст к данной работе и подтверждает зазор «forward-PINN без данных». Целевая статья, DNN-предшественник (2022) и диссертация Virginia Tech — фактически **одна научная группа** (Buchanan, Duarte; данные стенда KATHY).

| Работа | Сеть / данные | Что делает с DWO | Отношение к нашей задаче |
|---|---|---|---|
| **target:** Buchanan et al., «A recurrent neural network for modeling natural circulation density wave instabilities» (NET 2024) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1738573324006582) | LSTM/RNN; временные ряды DWO (натуральная циркуляция) | Моделирует/прогнозирует колебания расхода (амплитуда, динамика) | Тот же объект (DWO), но **data-driven surrogate**, не решение уравнений. Прямой антипод по философии — «учим по данным» vs «решаем физику без данных». |
| «Prediction of Unstable Two-Phase Flow Behavior using Dense Neural Network» (2022) — [ResearchGate](https://www.researchgate.net/publication/366016952_Prediction_of_Unstable_Two-Phase_Flow_Behavior_using_Dense_Neural_Network) | Dense NN (DNN); та же группа (Buchanan, VT) | Предсказывает амплитуду flow-oscillation / границу DWI (Type-II) | Предшественник target-статьи; DNN хуже ловит динамику → мотивация перехода к рекуррентным. |
| Hurley (Virginia Tech, дисс.) «Density-Wave Instability Characterization in BWRs under MELLLA+ during ATWS» — [VTechWorks](https://vtechworks.lib.vt.edu/items/02ff98f2-1538-498b-a7e3-d6d17abd8347) | 2 NN-модели на эксп. данных KATHY + TRACE/point-kinetics | Карта устойчивости / параметрика DWO; ML vs физ.модель | Источник данных и контекста для target/DNN-работ; явно фиксирует trade-off «сложность ↔ физичность» surrogate. |
| «Classification of two-phase flow instability phases using convolutional neural networks» (INIS/IAEA) — [INIS](https://inis.iaea.org/records/wf4a5-rsx66) | CNN, классификация по изображениям режимов потока | Распознаёт фазы неустойчивости в натуральной циркуляции | NN на DWO, но задача — классификация картинок, не динамика полей. Другой угол. |
| Gupta et al. «Numerical simulation and artificial neural network modeling of natural circulation boiling water reactor» (Nucl. Eng. Des. 2007) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029549306004304) | ANN-суррогат, обучен на RELAP5 | Быстрая параметрика устойчивости NC-BWR | Ранний пример NN-суррогата устойчивости кипящего канала; полностью data-driven. |
| Lombardi et al. «Prediction of two-phase mixture density using artificial neural networks» (Ann. Nucl. Energy 1997) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306454997000066) | ANN, плотность кипящей смеси | Не DWO напрямую — плотность смеси (вход в нейтронику/динамику) | Историческая отправная точка NN в этой нише; косвенно. |

**Зазор по сверхкритике.** В сверхкритическом домене нейросети применяют почти исключительно к **теплоотдаче / HTD** (предсказание Tw, Nu), а не к неустойчивости течения. NN именно для **DWO-в-сверхкритике** в выдаче не нашлось → ниша данной работы пуста даже в data-driven постановке, не только в forward-PINN.

**Вывод для позиционирования.** Весь существующий NN-кластер по DWO — supervised (RNN/DNN/CNN/ANN на данных кода или стенда). Forward-PINN, который достаёт DWO из уравнений **без supervised-данных**, не пересекается ни с одной из этих работ ни по методу, ни (для сверхкритики) по объекту. Цитировать этот кластер как «related ML work — все data-driven» в контрасте с предлагаемым подходом.

---

## Соседи по физике (traditional solvers — для постановки задачи, не метода)

DWO в сверхкритике классически решают time-domain implicit FD / coupled neutronic-TH, **не** ML:

- Numerical analysis of DWO + heat transfer deterioration in SCWR — [Springer](https://link.springer.com/article/10.1007/s12206-018-0208-7) / [academia PDF](https://www.academia.edu/129251529/Numerical_analysis_of_density_wave_instability_and_heat_transfer_deterioration_in_a_supercritical_water_reactor)
- Nonlinear analysis: subcritical / **supercritical / generalized Hopf** bifurcations + first Lyapunov coefficients, limit-cycle behaviour — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0149197021000111)
- Nuclear-coupled TH parallel-channel DWO в SCWR — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306454922005254)
- The analysis of density wave instability of supercritical water in two parallel channels — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306454920307106)
- Nonlinear coupled neutronic–thermohydraulic stability of SCWR — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306454923005169)
- NN-работы по DWO (LSTM/DNN/CNN/ANN, все data-driven) вынесены в отдельный раздел «Нейросети любого типа для DWO» выше — это целевой DOI 10.1016/j.net.2024.103407 и его соседи

Соседние instability-PINN для архитектурных приёмов:

- **KH-PINN** (Kelvin–Helmholtz, variable density) — [arXiv:2411.07524](https://arxiv.org/pdf/2411.07524)
- Shock-front benchmarking (почему vanilla-PINN мажет резкие фронты) — [arXiv:2503.17379](https://arxiv.org/pdf/2503.17379)
- Extended PINN для гиперболического two-phase flow — [arXiv:2511.13734](https://arxiv.org/html/2511.13734v2)
- TL-PINN для transients ядерного реактора — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10558465/)

---

## Три гипотезы для работы

1. **Причинность — необходимое условие.** Гладкое QS-решение неустойчиво; NTK-bias тянет PINN именно на него. Causal weighting (Wang+2024) — документированный способ заставить нестабильность вырасти. → `causal_eps`.
2. **Достижимость ≠ существование.** Dong+ показывают: предельный цикл *существует* как решение NS, но time-stepping его не достигает, а forward-PINN — да, при правильной инициализации. → оправдывает warm-start + gate как «выбор ветки решения», а не data-fitting.
3. **Нормировка невязки определяет, какую ветку поймаешь.** Абсолютный residual давит тонкую неустойчивость (arXiv:2511.22819). → per-zone keepers; сравнить с gradient-normalized residual как более принципиальной альтернативой ручным весам.

---

## Возможный следующий шаг

Глубокий разбор любой из двух ядровых работ (Wang+2024 или Dong+): вытащить точную формулу causal-веса / схему инициализации и сверить 1:1 с тем, что стоит в ноутбуке `pinn_v3b_imposed_dp.ipynb`.
