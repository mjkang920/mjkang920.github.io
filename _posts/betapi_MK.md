<h4 style="font-family:Arial; font-weight:bold; font-size:20px;">Initialization and Parameter Setting</h4>

<div style="font-family: Arial, font-size: 17px;">
The state of the five edges are represented by component events.  
The edges take binary-state, 0 for non-functional and 1 for functional.
</div>

probs = {'e1': {0: 0.01, 1: 0.99}, 'e2': {0: 0.01, 1: 0.99}, 'e3': {0: 0.05, 1: 0.95},
         'e4': {0: 0.05, 1: 0.95}, 'e5': {0: 0.10, 1: 0.90}}


```python
import itertools
import numpy as np
import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb 

# Define the network
nodes = {'n1': (0, 0),
         'n2': (1, 1),
         'n3': (1, -1),
         'n4': (2, 0)}

edges = {'e1': ['n1', 'n2'],
         'e2': ['n1', 'n3'],
         'e3': ['n2', 'n3'],
         'e4': ['n2', 'n4'],
         'e5': ['n3', 'n4']}

probs = {'e1': 0.99, 'e2': 0.99, 'e3': 0.95, 'e4': 0.95, 'e5': 0.90}  # 엣지 신뢰도(survival probability)
```

<h4 style="font-family:Arial; font-weight:bold; font-size:24px;">
Functions for Reliability Index and Redundancy Index
</h4>

$$
\beta_{ij} = -\Phi^{-1}(P(F_i \mid H_j))
$$

$$
\pi_{ij} = -\Phi^{-1}(P(F_{\text{sys}} \mid F_i, H_j))
$$

#### Flowchart: Monte Carlo Sampling and Redundancy Analysis

1. **가능한 모든 failure scenario 생성**
   - 모든 가능한 고장 시나리오를 MECE 방식으로 생성합니다.

2. **Monte Carlo Sampling**
   - 무작위로 샘플링된 시나리오 평가.
   - Sample과 MECE 시나리오를 비교하여 시스템 상태를 확인합니다.

3. **Net_conn 함수로 시스템 상태 평가**
   - **Case 1:** 시스템 상태가 실패(`System Fail`)일 경우 → `Pi Count` 증가.
   - **Case 2:** 시스템 상태가 생존(`System Survive`)일 경우 → 다음 단계 진행.

4. **While 조건 반복 (Fail 부재 수 < 총 부재 수)**
   - 생존한 부재에 대해 다음 확률을 부여:
     - `Survive`: 0.8
     - `Fail`: 0.2
   - 고장이 추가 발생하면(`tmp_failure2 > tmp_failure`) 시스템 상태를 재평가.

5. **System state 재평가**
   - 추가 고장이 없다면(`추가 부재 파손 X`), 루프를 종료 (`break`).
   - 시스템 상태가 `Fail`이면 `Pi Count` 증가.

6. **결과**
   - `Pi Count` 및 시스템 상태 결과 저장.


%%html
<script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs";
    mermaid.initialize({ startOnLoad: true });
</script>
<div class="mermaid">
graph TD
    A[Node A] --> B[Node B]
    B --> C[Node C]
    C --> D[Node D]
</div>




```python
def calculate_beta_pi(edges, probs, od_pair, MCS_N=1000000):
    num_edges = len(edges)
    num_mece = 2 ** num_edges - 1  
    mece = np.array([list(map(int, f"{i:0{num_edges}b}")) for i in range(1, 2 ** num_edges)])  

    # Count variables for each MECE scenario
    beta_count = np.zeros(num_mece)
    pi_count = np.zeros(num_mece)

    # Monte Carlo simulation
    for _ in range(MCS_N):
        comps_st = {e: np.random.choice([0, 1], p=[1 - probs[e], probs[e]]) for e in edges}  # 0: fail, 1: survive - state
        failure_mask = np.array([1 if comps_st[e] == 0 else 0 for e in edges])  # 1: fail, 0: survive - mask

        for scn_idx in range(num_mece):
            if np.array_equal(mece[scn_idx], failure_mask):
                beta_count[scn_idx] += 1

                f_val, sys_st, _ = net_conn(comps_st, od_pair, edges)
                if sys_st == 'f':  # If system is already failed
                    pi_count[scn_idx] += 1
                    break  # Skip load redistribution

                tmp_failure = np.sum(failure_mask)
                while tmp_failure < num_edges:
                    for e, state in comps_st.items():
                        if state == 1:  # if survive
                            comps_st[e] = np.random.choice([0, 1], p=[0.2, 0.8])

                    new_mask = np.array([1 if comps_st[e] == 0 else 0 for e in edges])
                    tmp_failure2 = np.sum(new_mask)

                    if tmp_failure2 > tmp_failure:  # New failures detected
                        f_val, sys_st, _ = net_conn(comps_st, od_pair, edges)
                        if sys_st == 'f':  # If system is in fail state
                            pi_count[scn_idx] += 1
                            break

                    if tmp_failure2 == tmp_failure:
                        break
                    tmp_failure = tmp_failure2

    beta_count[beta_count == 0] = 1e-5
    pi_count[pi_count == 0] = 1e-5
    redundancy = pi_count / beta_count
    redundancy = np.clip(redundancy, 1e-10, 1 - 1e-10)
    reliability = beta_count / MCS_N

    beta = -norm.ppf(reliability)
    pi = -norm.ppf(redundancy)

    beta[beta < -3] = -3
    pi[pi < -3] = -3

    # Print beta_count and pi_count for each MECE case
    print("\nMECE Case | Beta Count | Pi Count")
    print("---------------------------------")
    for idx, case in enumerate(mece):
        print(f"{case} | {beta_count[idx]:.5f} | {pi_count[idx]:.5f}")

    return beta, pi, mece

```

<h4 style="font-family:Arial; font-weight:bold; font-size:20px;">
Network Connectivity Evaluation Function
</h4>


```python
def net_conn(comps_st, od_pair, edges):
    G = nx.Graph()
    for k, state in comps_st.items():
        if state == 1:  # if survive
            G.add_edge(edges[k][0], edges[k][1], capacity=1)  # capacity 직접 설정

    # OD 쌍 목적지에 새로운 노드 추가
    G.add_edge(od_pair[1], 'new_d', capacity=1)

    try:
        f_val, f_dict = nx.maximum_flow(G, od_pair[0], 'new_d', capacity='capacity', flow_func=nx.algorithms.flow.shortest_augmenting_path)

        if f_val > 0:
            sys_st = 's'
            return f_val, sys_st, {k: state for k, state in comps_st.items() if state == 1}
        else:
            return f_val, 'f', None
    except Exception as e:
        return 0, 'f', None
```

<h4 style="font-family:Arial; font-weight:bold; font-size:20px;">
Results
</h4>


```python
# Main execution
beta_values = calculate_reliability(edges, probs)
beta, pi, mece = calculate_beta_pi(edges, probs, od_pair=('n1', 'n4'), MCS_N=10000000)

# Results output
print("Scenario | MECE Subset | Beta | Pi")
print("----------------------------------------------------")
for scn_idx in range(len(mece)):
    print(f"Scenario {scn_idx + 1}: {mece[scn_idx]}  -->  Beta: {beta[scn_idx]:.6f} | Pi: {pi[scn_idx]:.6f}")

# Count the number of failed elements for each MECE subset
failed_elements = np.sum(mece, axis=1)  # 실패한 부재 수 계산

# Define distinct colors for each number of failed elements
unique_fail_counts = np.unique(failed_elements)  # 고유한 실패 부재 개수
dominant_colors_hex = ['#f28b82', '#fbbc04', '#fff475', '#ccff90', '#aecbfa', '#d7aefb']
colors = [to_rgb(color) for color in dominant_colors_hex]

# Visualization
plt.figure(figsize=(8, 8))  # 전체 플롯 크기 설정

# 1. Scatter plot for β-π Diagram
for count, color in zip(unique_fail_counts, colors):
    indices = failed_elements == count  # 현재 실패 부재 개수와 일치하는 시나리오
    plt.scatter(pi[indices], beta[indices], color=color, s=200,  # 점의 크기 조정 (s=100)
                label=f"{count} Element Fail", edgecolors='k', alpha=0.7)

# 2. Plot Threshold Lines for Different prob_target Values
prob_targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
reverse_labels = [r'$\lambda_{H_j} = 10^{-5}$', r'$\lambda_{H_j} = 10^{-4}$', 
                  r'$\lambda_{H_j} = 10^{-3}$', r'$\lambda_{H_j} = 10^{-2}$', r'$\lambda_{H_j} = 10^{-1}$']
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # 다양한 선 스타일 설정

for prob_target, line_style, label in zip(prob_targets, line_styles, reverse_labels):
    beta_threshold = np.arange(-3, 8, 1e-4)  # β threshold range
    pi_threshold = -norm.ppf(prob_target / norm.cdf(-beta_threshold))  # π values for given threshold
    plt.plot(pi_threshold, beta_threshold, 'indianred', linestyle=line_style, 
             label=label, alpha=0.8)  # 포맷 변경

# Customize the plot
plt.title(r'$\beta - \pi$ Diagram with Thresholds', fontsize=16)
plt.xlabel('Redundancy (π)', fontsize=12)
plt.ylabel('Reliability (β)', fontsize=12)
plt.axis('equal')
plt.xlim([-3, 7])
plt.ylim([-3, 7])
plt.grid(alpha=0.5)
plt.legend(title="Failure Scenarios and Thresholds", fontsize=10, loc="best")
plt.tight_layout()
plt.show()
```

    
    MECE Case | Beta Count | Pi Count
    ---------------------------------
    [0 0 0 0 1] | 885166.00000 | 353817.00000
    [0 0 0 1 0] | 419216.00000 | 167511.00000
    [0 0 0 1 1] | 46487.00000 | 46487.00000
    [0 0 1 0 0] | 417811.00000 | 131306.00000
    [0 0 1 0 1] | 46134.00000 | 18716.00000
    [0 0 1 1 0] | 22144.00000 | 9026.00000
    [0 0 1 1 1] | 2388.00000 | 2388.00000
    [0 1 0 0 0] | 80440.00000 | 32216.00000
    [0 1 0 0 1] | 8949.00000 | 3619.00000
    [0 1 0 1 0] | 4262.00000 | 2100.00000
    [0 1 0 1 1] | 465.00000 | 465.00000
    [0 1 1 0 0] | 4390.00000 | 1849.00000
    [0 1 1 0 1] | 467.00000 | 169.00000
    [0 1 1 1 0] | 238.00000 | 238.00000
    [0 1 1 1 1] | 33.00000 | 33.00000
    [1 0 0 0 0] | 80420.00000 | 32220.00000
    [1 0 0 0 1] | 8806.00000 | 4365.00000
    [1 0 0 1 0] | 4274.00000 | 1766.00000
    [1 0 0 1 1] | 468.00000 | 468.00000
    [1 0 1 0 0] | 4179.00000 | 1735.00000
    [1 0 1 0 1] | 505.00000 | 505.00000
    [1 0 1 1 0] | 223.00000 | 77.00000
    [1 0 1 1 1] | 24.00000 | 24.00000
    [1 1 0 0 0] | 838.00000 | 838.00000
    [1 1 0 0 1] | 74.00000 | 74.00000
    [1 1 0 1 0] | 38.00000 | 38.00000
    [1 1 0 1 1] | 4.00000 | 4.00000
    [1 1 1 0 0] | 42.00000 | 42.00000
    [1 1 1 0 1] | 10.00000 | 10.00000
    [1 1 1 1 0] | 1.00000 | 1.00000
    [1 1 1 1 1] | 0.00001 | 0.00001
    Scenario | MECE Subset | Beta | Pi
    ----------------------------------------------------
    Scenario 1: [0 0 0 0 1]  -->  Beta: 1.349946 | Pi: 0.254076
    Scenario 2: [0 0 0 1 0]  -->  Beta: 1.728809 | Pi: 0.254430
    Scenario 3: [0 0 0 1 1]  -->  Beta: 2.600920 | Pi: -3.000000
    Scenario 4: [0 0 1 0 0]  -->  Beta: 1.730381 | Pi: 0.483779
    Scenario 5: [0 0 1 0 1]  -->  Beta: 2.603534 | Pi: 0.238652
    Scenario 6: [0 0 1 1 0]  -->  Beta: 2.845886 | Pi: 0.233711
    Scenario 7: [0 0 1 1 1]  -->  Beta: 3.493015 | Pi: -3.000000
    Scenario 8: [0 1 0 0 0]  -->  Beta: 2.406913 | Pi: 0.252060
    Scenario 9: [0 1 0 0 1]  -->  Beta: 3.123062 | Pi: 0.241967
    Scenario 10: [0 1 0 1 0]  -->  Beta: 3.335194 | Pi: 0.018233
    Scenario 11: [0 1 0 1 1]  -->  Beta: 3.908164 | Pi: -3.000000
    Scenario 12: [0 1 1 0 0]  -->  Beta: 3.326957 | Pi: 0.198864
    Scenario 13: [0 1 1 0 1]  -->  Beta: 3.907127 | Pi: 0.353426
    Scenario 14: [0 1 1 1 0]  -->  Beta: 4.067109 | Pi: -3.000000
    Scenario 15: [0 1 1 1 1]  -->  Beta: 4.506196 | Pi: -3.000000
    Scenario 16: [1 0 0 0 0]  -->  Beta: 2.407004 | Pi: 0.251674
    Scenario 17: [1 0 0 0 1]  -->  Beta: 3.127800 | Pi: 0.010817
    Scenario 18: [1 0 0 1 0]  -->  Beta: 3.334412 | Pi: 0.219331
    Scenario 19: [1 0 0 1 1]  -->  Beta: 3.906610 | Pi: -3.000000
    Scenario 20: [1 0 1 0 0]  -->  Beta: 3.340658 | Pi: 0.214263
    Scenario 21: [1 0 1 0 1]  -->  Beta: 3.888177 | Pi: -3.000000
    Scenario 22: [1 0 1 1 0]  -->  Beta: 4.082259 | Pi: 0.398064
    Scenario 23: [1 0 1 1 1]  -->  Beta: 4.573344 | Pi: -3.000000
    Scenario 24: [1 1 0 0 0]  -->  Beta: 3.763428 | Pi: -3.000000
    Scenario 25: [1 1 0 0 1]  -->  Beta: 4.331644 | Pi: -3.000000
    Scenario 26: [1 1 0 1 0]  -->  Beta: 4.476153 | Pi: -3.000000
    Scenario 27: [1 1 0 1 1]  -->  Beta: 4.935367 | Pi: -3.000000
    Scenario 28: [1 1 1 0 0]  -->  Beta: 4.454727 | Pi: -3.000000
    Scenario 29: [1 1 1 0 1]  -->  Beta: 4.753424 | Pi: -3.000000
    Scenario 30: [1 1 1 1 0]  -->  Beta: 5.199338 | Pi: -3.000000
    Scenario 31: [1 1 1 1 1]  -->  Beta: 7.034484 | Pi: -3.000000
    


    
![png](betapi_MK_files/betapi_MK_9_1.png)
    



