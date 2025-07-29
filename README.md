# **Deep Q-Learning (DQL) Snake AI**

---

### **1. Introduction**
This project implements a Deep Q-Learning (DQL) agent that learns to play the classic Snake game. The agent is trained using a neural network (Q-Network) built with PyTorch, incorporating key DQL techniques such as experience replay and a target network for stable learning. The training process is visualized in real-time using a multi-window Matplotlib dashboard, providing insights into various performance and learning metrics.

---

### **2. Features**
* **Deep Q-Learning (DQL):** Employs a robust DQL algorithm for the AI agent.
* **PyTorch Implementation:** Neural network (Q-Network) and training logic built using PyTorch.
* **Experience Replay Buffer:** Utilizes a `deque` (double-ended queue) for storing and sampling past experiences to break correlations and improve learning stability.
* **Target Network:** Incorporates a separate target Q-network to stabilize Q-value targets during training.
* **Reward Shaping:** Enhanced reward function that encourages the snake to move towards food and penalizes moving away, in addition to standard food/collision rewards.
* **Live Matplotlib Dashboard:** Visualizes training progress in real-time across multiple metrics:
    * Game Scores & Mean Score
    * Average Loss per Episode
    * Average Max Q-Value
    * Episode Length (Steps)
    * Efficiency (Score per 100 Steps)
* **Model Saving/Loading:** Automatically saves the best-performing models and replay memory, allowing for training resumption.
* **Reproducibility:** Includes a helper function to set random seeds for consistent results.

---

### **3. Technologies Used**
This project leverages the following key technologies:

* **Programming Language:**
    * Python 3.x
* **Core Libraries/Frameworks:**
    * **PyTorch:** For building, training, and running the deep neural networks.
    * **Pygame:** Used for rendering the Snake game environment.
    * **NumPy:** For numerical operations and state representation.
    * **Matplotlib:** For generating the live multi-window training dashboard.
    * **`collections.deque` (Python Built-in):** Used for the experience replay buffer.
    * **`pickle` (Python Built-in):** Used for saving and loading the replay memory.

---

### **4. Getting Started**
Follow these instructions to set up and run the project on your local machine.

#### **4.1. Prerequisites**
Ensure you have the following software installed:

* **Python 3.x:** Download from [python.org](https://www.python.org/). It's recommended to use Python 3.8 or newer.
* **Git:** Download from [git-scm.com](https://git-scm.com/downloads).

#### **4.2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
    (Replace `[your-username]` and `[your-repo-name]` with your actual GitHub details, e.g., `snake-dql-ai`)

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **For GPU acceleration (if you have an NVIDIA GPU):**
        By default, `torch` from `requirements.txt` might install the CPU-only version. If you want GPU support, you might need to install PyTorch manually from their official website (pytorch.org) using their specific installation commands that match your CUDA version, then run `pip install -r requirements.txt` again for other dependencies.

---

### **5. Usage**

To start the DQL agent training and visualize its progress:

1.  **Activate your virtual environment** (if not already active):
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

2.  **Navigate to the root of the project directory:**
    ```bash
    cd [your-repo-name]
    ```

3.  **Start Training:**
    ```bash
    python train.py
    ```
    * This will open a Pygame window showing the Snake game, and three separate Matplotlib windows (the Dashboard) displaying live plots of scores, losses, Q-values, episode lengths, and efficiencies.
    * Training progress will also be logged in your terminal.
    * The best models and replay memory will be saved in the `model_weights/` directory automatically.

4.  **Resume Training (Optional):**
    If you want to continue training from where you left off (using previously saved models and replay memory):
    ```bash
    python train.py --resume
    ```

5.  **Specify a Random Seed (Optional, for Reproducibility):**
    ```bash
    python train.py --seed 123
    ```

---

### **6. Screenshots/Demos**
*(Replace these with actual links to your images or GIFs)*

* **Snake Game AI in Action:** A GIF or screenshot of the trained AI playing the Snake game.
    ![Snake Game AI Demo](https://via.placeholder.com/700x400?text=Snake+AI+Playing)

* **Live Matplotlib Dashboard (Scores):** A screenshot showing the "Game Scores & Mean Score" plot.
    ![Scores Dashboard](https://via.placeholder.com/700x400?text=Matplotlib+Scores+Plot)

* **Live Matplotlib Dashboard (Learning Metrics):** A screenshot showing the "Learning Process Metrics" plot (Average Loss & Average Max Q-Value).
    ![Learning Metrics Dashboard](https://via.placeholder.com/700x400?text=Matplotlib+Learning+Metrics)

* **Live Matplotlib Dashboard (Behavioral Metrics):** A screenshot showing the "Behavioral Metrics" plot (Episode Length & Efficiency).
    ![Behavioral Metrics Dashboard](https://via.placeholder.com/700x400?text=Matplotlib+Behavioral+Metrics)

---

### **7. Project Structure**

*(Note: Excluded large model weights, replay memory, and virtual environments from this structure for clarity on GitHub.)*

---

### **8. Contributing**
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

### **9. License**
This project is licensed under the **MIT License** - see the `LICENSE` file for details. (Create a `LICENSE` file in the root of your project if you don't have one).

---

### **10. Contact**
[Your Name] - [your.email@example.com]
Project Link: [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name)

---
