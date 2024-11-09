import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import copy

blocker = None


def get_expert_predictions(beautiful_soup_thing) -> int:
    """
    Get expert prediction from Trendlyne. Returns None if no rating found
    Simply takes the first rating it finds.
    """
    for list in beautiful_soup_thing:
        for inner_list in list:
            for word in inner_list:
                if word in ['Buy', 'buy', 'Bullish', 'bullish', 'Strong Buy', 'strong buy', 'Outperform', 'outperform']:
                    return 1
                elif word in ['Hold', 'hold', 'Neutral', 'neutral', 'Market Perform', 'market perform']:
                    return 0
                elif word in ['Sell', 'sell', 'Bearish', 'bearish', 'Strong Sell', 'strong sell', 'Underperform', 'underperform']:
                    return -1
    return None

def get_sentiment_predictions(ticker: str) -> list[float]:
    """Gets the scores of the several aspects in the ABSA Done by Pratyush and Sirjan"""
    global blocker

    aspectal_params = blocker[ticker]

    return np.array(aspectal_params.values())

def get_stock_price(ticker: str) -> float:
    """
    Get the current stock price for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Current stock price
    """
    return yf.Ticker(ticker).history(period='1d')['Close'].iloc[0]

class MultiplicativeWeightsExpert:
    def __init__(self, stock: str, experts_list: list[str], params: dict, learning_rate: float = 0.1, movement_threshold: float = 0.01, verdict_threshold=0.05):
        """
        Initialize the multiplicative weights expert model.
        
        Args:
            stock (str): Stock ticker
            experts_list (List[str]): List of expert names
            learning_rate (float): Learning rate for weight updates
            movement_threshold (float): Threshold for stock movement
        """
        global blocker
        self.stock = stock
        self.experts = experts_list
        self.num_experts = len(experts_list)
        self.learning_rate = learning_rate
        self.movement_threshold = movement_threshold
        self.verdict_threshold = verdict_threshold
        blocker = copy.deepcopy(params)
        # Initialize weights uniformly
        self.weights = np.ones(self.num_experts) / self.num_experts
        
        # Track losses for each expert
        self.losses = np.zeros(self.num_experts)    
        self.cumulative_losses = np.zeros(self.num_experts)
        self.total_loss = 0
        self.weights_history = np.zeros((0, self.num_experts))
        self.expert_losses_history = np.zeros((0, self.num_experts))
        self.total_loss_history = []

        # Track predictions
        self.current_expert_predictions = np.zeros(self.num_experts)
        self.model_rating = 0
        self.verdict = 0    # Buy = 1, Hold = 0, Sell = -1
        
    def get_prediction(self, expert_predictions: list[float]) -> float:
        """
        Get weighted average of expert predictions.
        
        Args:
            expert_predictions (List[float]): List of predictions from each expert
            
        Returns:
            float: Weighted average prediction
        """
        if len(expert_predictions) != self.num_experts:
            raise ValueError("Number of predictions must match number of experts")
               
        return np.dot(self.weights, expert_predictions)
    
    def get_verdict(self, rating: float) -> int:
        if rating > self.verdict_threshold:
            return 1
        if rating < -self.verdict_threshold:
            return -1
        return 0
    
    def calculate_expert_losses(self, expert_predictions: list[float], actual_movement: int) -> np.ndarray:
        """
        Calculate losses for each expert based on their predictions.
        
        Args:
            expert_predictions (List[float]): List of predictions from each expert
            actual_movement (int): 1 if stock increased, -1 if decreased
            
        Returns:
            np.ndarray: Array of losses for each expert
        """
        if actual_movement > self.movement_threshold:
            return np.array([(1 - pred)**2 for pred in expert_predictions])
        elif actual_movement < -self.movement_threshold:
            return np.array([(1 + pred)**2 for pred in expert_predictions])
        else:
            return np.array([(pred)**2 for pred in expert_predictions])
    
    def update_weights(self, expert_losses: np.ndarray):
        """
        Update weights based on expert losses.
        
        Args:
            expert_losses (np.ndarray): Array of losses for each expert
        """
        # Update weights using exponential punishment
        self.weights = self.weights * np.exp(-self.learning_rate * expert_losses)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        # Update cumulative losses
        self.cumulative_losses += expert_losses
        self.total_loss += np.dot(self.weights, expert_losses)

        # Save history
        self.weights_history = np.vstack((self.weights_history, self.weights))

    def compute_movement(self, x: float, clipper = lambda x: 1/(1 + np.exp(-x/0.2))) -> int:
        """
        Compute movement based on a sigmoid function.
        
        Args:
            clipper (lambda): Sigmoid function
            x (float): Input value
            
        Returns:
            int: 1 if stock increased, -1 if decreased, smoothed by the clipper
        """
        return clipper(x)
    
    def forward(self, movement: int, clipper=lambda x: 1/(1 + np.exp(-x/0.2))) -> float:
        """
        Forward pass of the model. Computes loss, updates weights and history
        Args: movement (change in stock value (ratio) between -1 to 1)
        Returns: prediction of the model
        """
        # Note, this runs every night

        modified_movement = self.compute_movement(clipper, movement)
        expert_predictions = self.current_expert_predictions
        expert_losses = self.calculate_expert_losses(expert_predictions, modified_movement)
        self.update_weights(expert_losses)

        # Save history
        self.expert_losses_history = np.vstack((self.expert_losses_history, expert_losses))

        # Update the predictions, for the next day
        self.current_expert_predictions = get_sentiment_predictions(self.stock) + [get_expert_predictions(self.stock)]
        self.model_rating = self.get_prediction(self.current_expert_predictions)
        self.verdict = self.get_verdict(self.model_rating)


    def get_parameters(self) -> dict:
        """
        Get current results and statistics.
        
        Returns:
            Dict: Dictionary containing current weights, losses, and other statistics
        """
        return {
            'weights': self.weights,
            # 'losses': self.losses,
            # 'cumulative_losses': self.cumulative_losses,
            'current_predictions': self.current_expert_predictions,
        }

# # There has to be a model per each stock
# for ticker in database:
#     # make model
#     model = MultiplicativeWeightsExpert(ticker)


# To get weights (you will get the weights in the order of your aspects, trendlyne at the end)
# model.weights

# to get model rating
# model.model_rating

# to get the verdict
# model.verdict

# to do a forward pass (I don't think you need this)
# model.forward(movement)

# Movement is the growth = (final_price - inital_price)/(initial_price)
