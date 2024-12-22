import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple, Optional, Union, Callable, List, Literal


class OptunaObjective:
    def __init__(
        self,
        model_class,
        metric: Callable,
        hyperparam_space: Dict[str, Callable[[optuna.Trial], Any]],
        fixed_params_space: Dict[str, Any] = {},
        random_state: int = 42,
        # Validation parameters
        validation_strategy: Literal['cv', 'validation', 'xgboost_val'] = 'xgboost_val',
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        n_trials: int = 100,
        optimization_direction: List[str] = 'minimize',
        return_model: bool = False
    ) -> None:
        
        """
        Initialize the OptunaObjective with model parameters and validation setup.

        Parameters
        ----------
        model_class : class
            The class of the model to be optimized.
        metric : Callable
            A callable function to evaluate the model's performance.
        hyperparam_space : Dict[str, Callable[[optuna.Trial], Any]]
            A dictionary defining the hyperparameter search space, with each entry being 
            a parameter name and a callable that returns a sampled value given an Optuna trial.
        fixed_params_space : Dict[str, Any], optional
            A dictionary of fixed parameters for the model. Default is an empty dictionary.
        random_state : int, optional
            Seed for random number generation to ensure reproducibility. Default is 42.
        validation_strategy : {'cv', 'validation', 'xgboost_val'}, optional
            The strategy to be used for validation. Default is 'xgboost_val'.
        X_train : np.ndarray, optional
            Training data features. Required for 'cv' and 'validation' strategies.
        y_train : np.ndarray, optional
            Training data labels. Required for 'cv' and 'validation' strategies.
        X_val : np.ndarray, optional
            Validation data features. Required for 'validation' strategy.
        y_val : np.ndarray, optional
            Validation data labels. Required for 'validation' strategy.
        cv_folds : int, optional
            Number of cross-validation folds. Default is 5.
        n_trials : int, optional
            Number of trials for the optimization process. Default is 100.
        optimization_direction : List[str], optional
            The direction of optimization, either 'minimize' or 'maximize'. Default is 'minimize'.
        return_model : bool, optional
            Whether to return the best model after optimization. Default is False.
        """
        
        self.model_class = model_class
        self.metric = metric
        self.hyperparam_space = hyperparam_space
        self.fixed_params_space = fixed_params_space
        self.random_state = random_state
        self.validation_strategy = validation_strategy
        self.n_trials = n_trials
        self.direction = optimization_direction
        self.return_model = return_model
        self.best_validation_score = None
        
        # Validation data setup
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
        
        self._input_validation()
        self._best_model_return_check()
    
    
    def _input_validation(self) -> None:
        """
        Validate inputs based on the chosen validation strategy.

        Validation strategy can be one of the following:
        - 'cv': Cross-validation. Requires `X_train` and `y_train`.
        - 'validation': Validation split. Requires `X_train`, `y_train`, `X_val` and `y_val`.
        - 'xgboost_val': XGBoost validation. No additional inputs required.
        """
        
        if self.validation_strategy == 'cv':
            if self.X_train is None or self.y_train is None:
                raise ValueError("X_train and y_train are required for cross-validation")
        elif self.validation_strategy == 'validation':
            if any(v is None for v in [self.X_train, self.y_train, self.X_val, self.y_val]):
                raise ValueError("Both training and validation data are required for validation split strategy")
        elif self.validation_strategy == 'xgboost_val':
            pass
        else:
            raise ValueError("validation_strategy must be either 'cv', 'validation' or 'xgboost_val'")
        
        
    def __cross_validation_metric(self, estimator, X, y) -> float:
        """
        Calculate cross-validation metric.

        Parameters
        ----------
        estimator : object
            The model to evaluate.
        X : object
            The features to use for prediction.
        y : object
            The target variable.

        Returns
        -------
        score : float
            The calculated metric.
        """
        
        # Generate predictions
        predictions = estimator.predict(X)
        
        # Calculate metric
        score = self.metric(y, predictions)
        return score
    
    def _best_model_return_check(self) -> None:
        """
        Initialize best_validation_score based on optimization direction.
        """
        
        if self.return_model:
            if self.direction == 'minimize':
                self.best_validation_score = np.inf
            elif self.direction == 'maximize':
                self.best_validation_score = -np.inf
        
    def evaluate_model(self, model) -> float:
        """
        Evaluate model based on chosen validation strategy.
        
        Args:
            model: Fitted model to evaluate
            
        Returns:
            float: Evaluation metric score
        """
        
        if self.validation_strategy == 'cv':
            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                scoring=self.__cross_validation_metric,
                cv=self.cv_folds
            )
            score = np.mean(scores)
        elif self.validation_strategy == 'validation':
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_val)
            score = self.metric(self.y_val, predictions)
        
        return score
        

    def objective(self, trial: optuna.Trial) -> float:
        """
        Define the objective function to be optimized.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.

        Returns
        -------
        evaluation_score : float
            The evaluation score based on the chosen validation strategy.
        """
        
        # Get hyperparameters from trial
        hyperparameters = {
            name: sampling_function(trial)
            for name, sampling_function in self.hyperparam_space.items()
        }
        
        # Combine with fixed parameters
        all_parameters = {**hyperparameters, **self.fixed_params_space}

        # Initialize the model based on validation strategy
        if self.validation_strategy == 'xgboost_val':
            model = self.model_class(**all_parameters, early_stopping_rounds=20)
            model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
            y_pred = model.predict(self.X_val)
            evaluation_score = self.metric(self.y_val, y_pred)
            
        else:  # For 'cv' or 'validation' strategies
            try:
                model = self.model_class(**all_parameters, random_state=self.random_state)
            except TypeError:
                model = self.model_class(**all_parameters)
                
            evaluation_score = self.evaluate_model(model)
        
        # Update best model for return
        if self.return_model:
            if self.direction == "minimize" and evaluation_score < self.best_validation_score:
                self.best_model = model
            elif self.direction == "maximize" and evaluation_score > self.best_validation_score:
                self.best_model = model
        
        return evaluation_score
    
    def optimize(self, 
                 show_progress_bar: bool = False
    ) -> Tuple[Dict[str, Any], float, Any, optuna.Study]:
        """
        Perform hyperparameter optimization using Optuna. (Optuna performs bayesian optimization)

        Args:
            show_progress_bar: Whether to show a progress bar during optimization.

        Returns:
            A tuple containing the best hyperparameters, the best score, the best model, and the Optuna study.
        """
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=show_progress_bar)
        
        return study.best_params, study.best_value, self.best_model, study
