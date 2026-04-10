<a id="readme-top"></a>

<div align="center">
  <h2 align="center">OmniBehavior: Towards Real-world Human Behavior Simulation</h2>

  <p align="center">
    Benchmarking LLMs on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces
  </p>
</div>


<p align="center">
  🌐 <a href="https://omnibehavior.github.io/" target="_blank">Website</a> &nbsp; | &nbsp;
  📃 <a href="https://arxiv.org/abs/2604.08362" target="_blank">Paper</a> &nbsp; | &nbsp;
  🥳 <a href="#citation">Citation</a>
</p>


<p align="center">
  <img src="media/OmniBehavior.jpg" width="95%" />
</p>

## 🚀 What's New 🆕

- **[2026.05]** **Expected Release**: The complete dataset and evaluation code are expected to be released around May 2026 after a formal data auditing. Stay tuned! ✨
- **[2026.04.10]** We have released the **OmniBehavior** paper! Please check it out for more details on our comprehensive user behavior analysis. 🔥🔥🔥

## 🎯 Multiple Scenarios

**OmniBehavior** captures real user behaviors across several interactive scenarios in [Kuaishou](https://www.kuaishou.com/):

| Scene Type | Description |
| :--- | :--- |
| **Video Browsing**    | Behaviors related to browsing and watching short videos. |
| **Live Streaming**    | Interactions within live stream rooms (e.g., watching duration, comments, likes, gifts). |
| **E-commerce**        | Shopping activities such as browsing products, managing cart, and purchasing. |
| **Advertisement**     | User interactions with recommended advertisements (views, clicks, conversions). |
| **Customer Service**  | Chat logs and interactions with E-commerce customer service agents. |
| **Search Behavior**   | All in-app search activities, including but not limited to video and marketplace queries. |


## 📂 Demo Data 

`data/demo.json` provides a sample of the data format used in this project.

> **Note:**
> - This file contains a **partial subset** of data for a single user.
> - It is intended for demonstration and testing purposes only.
> - **The complete dataset will be made publicly available after a formal data auditing.**

### Dataset Highlights

This demo dataset represents a case study of a user in [Kuaishou](https://www.kuaishou.com/):

- **Long-term Observation**: The data spans **90 days** (from `2025-09-01` to `2025-11-30`), providing a substantial timeline to observe evolving user interests and habitual patterns.
- **Real Interaction**: The dataset contains lots of **real actions**, capturing a consistent and detailed trail of user interactions.
- **Comprehensive Scenario Coverage**: The schema supports capturing behavior across **mainstream short-video platform scenarios**.

### Case Study Value

Although `demo.json` showcases a single user, its depth makes it a valuable resource for research:
1.  **Long-term Interest Modeling**: The 3-month span allows for the specific tracking of interest shifts and stability over time.
2.  **Cross-Domain Behavior Analysis**: By covering diverse scenarios, it enables research into how behaviors in one domain (e.g., watching a streamer) correlate with actions in another (e.g., purchasing products or clicking ads).
3.  **User Behavior Simulation**: This detailed trajectory provides a ground truth for building user simulators, allowing to evaluate how well agents can simulate real, long-term human behavior patterns in complex environments.


<p align="center">
  <img src="media/multi_scenario.jpg" width="80%" />
</p>

## 🗂️ Data Structure

The data is organized by user ID. Each user entry contains a textual profile and a chronological history of actions.


```json
{
  "user_ID": {
    "user_profile": "Description of the user (e.g., demographics, education, etc.)...",
    "action_history": [
      {
        "type": "Scenario Type",
        "timestamp": "YYYY-MM-DD HH:MM:SS",
        "context": {
          "field_name": "value",
          ...
        },
        "action": [
          {
            "type": "specific_behavior",
            "attribute": "value"
          }
          ...
        ]
      },
      ...
    ]
  }
}
```

## 📜 License

This dataset is strictly prohibited for commercial use and is intended for **academic research purposes only**.



<a id="citation"></a>
## 📝 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{chen2026omnibehavior,
      title={Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces}, 
      author={Jiawei Chen and Ruoxi Xu and Boxi Cao and Ruotong Pan and Yunfei Zhang and Yifei Hu and Yong Du and Tingting Gao and Yaojie Lu and Yingfei Sun and Xianpei Han and Le Sun and Xiangyu Wu and Hongyu Lin},
      year={2026},
      eprint={2604.08362},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.08362}, 
}
```
