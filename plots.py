import matplotlib.pyplot as plt
import os

class Dashboard:
    """
    A class to manage a multi-window Matplotlib dashboard for live plotting.
    This class structure prevents warnings and ensures smooth updates by creating
    the figure windows only once during initialization.
    """
    def __init__(self):
        # Enable interactive mode. This is crucial for live updates.
        plt.ion()
        plt.style.use('fivethirtyeight')

        # --- Create all figure windows during initialization ---
        # This is the key change: figures are now created only ONCE.
        self.fig1 = plt.figure(1, figsize=(8, 6))
        self.fig2 = plt.figure(2, figsize=(12, 5))
        self.fig3 = plt.figure(3, figsize=(12, 5))

        # Set window titles once
        self.fig1.canvas.manager.set_window_title('Performance Metrics')
        self.fig2.canvas.manager.set_window_title('Learning Process Metrics')
        self.fig3.canvas.manager.set_window_title('Behavioral Metrics')

    def update(self, scores, mean_scores, avg_losses, episode_lengths, efficiencies, avg_q_values):
        """
        This method is called in the training loop to clear and redraw all plots
        with the latest data.
        """
        # --- Window 1: Primary Performance (Scores) ---
        self.fig1.clf() # Clear the figure to redraw it
        ax = self.fig1.add_subplot(111)
        ax.set_title('Game Scores & Mean Score', fontsize=14)
        ax.set_xlabel('Number of Games', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.plot(scores, label='Score per Game', color='dodgerblue', alpha=0.7)
        ax.plot(mean_scores, label='Mean Score', color='darkorange', linewidth=2, linestyle='--')
        ax.set_ylim(ymin=0)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if scores:
            ax.text(len(scores)-1, scores[-1], str(scores[-1]), ha='right', va='bottom', color='dodgerblue')
        if mean_scores:
            ax.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.2f}', ha='right', va='top', color='darkorange', weight='bold')

        # --- Window 2: Learning Process Metrics ---
        self.fig2.clf()
        self.fig2.suptitle('Learning Process Metrics', fontsize=16, weight='bold')
        ax1 = self.fig2.add_subplot(1, 2, 1)
        ax1.set_title('Average Loss per Episode', fontsize=12)
        ax1.set_xlabel('Number of Games', fontsize=10)
        ax1.set_ylabel('Loss (MSE)', fontsize=10)
        ax1.plot(avg_losses, color='crimson')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2 = self.fig2.add_subplot(1, 2, 2)
        ax2.set_title('Average Max Q-Value', fontsize=12)
        ax2.set_xlabel('Number of Games', fontsize=10)
        ax2.set_ylabel('Avg. Q-Value', fontsize=10)
        ax2.plot(avg_q_values, color='saddlebrown')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Window 3: Behavioral Metrics ---
        self.fig3.clf()
        self.fig3.suptitle('Behavioral Metrics', fontsize=16, weight='bold')
        ax3 = self.fig3.add_subplot(1, 2, 1)
        ax3.set_title('Episode Length (Steps)', fontsize=12)
        ax3.set_xlabel('Number of Games', fontsize=10)
        ax3.set_ylabel('Number of Steps', fontsize=10)
        ax3.plot(episode_lengths, color='purple')
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax4 = self.fig3.add_subplot(1, 2, 2)
        ax4.set_title('Efficiency (Score per 100 Steps)', fontsize=12)
        ax4.set_xlabel('Number of Games', fontsize=10)
        ax4.set_ylabel('Efficiency', fontsize=10)
        ax4.plot(efficiencies, color='green')
        ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # --- Final Step: Render the plots ---
        # plt.pause() draws the updates to all figures and processes GUI events.
        plt.pause(0.01)
