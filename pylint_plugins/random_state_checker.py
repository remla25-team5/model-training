from pylint.checkers import BaseChecker
import astroid

TARGETS_RANDOMNESS = {
    "StratifiedKFold": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"shuffle": True}
    },
    "KFold": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"shuffle": True}
    },
    "ShuffleSplit": {"type": "class", "seed_param": "random_state"},
    "StratifiedShuffleSplit": {"type": "class", "seed_param": "random_state"},
    "train_test_split": {"type": "function", "seed_param": "random_state"},

    "RandomForestClassifier": {"type": "class", "seed_param": "random_state"},
    "RandomForestRegressor": {"type": "class", "seed_param": "random_state"},
    "ExtraTreesClassifier": {"type": "class", "seed_param": "random_state"},
    "ExtraTreesRegressor": {"type": "class", "seed_param": "random_state"},
    "GradientBoostingClassifier": {"type": "class", "seed_param": "random_state"},
    "GradientBoostingRegressor": {"type": "class", "seed_param": "random_state"},
    "HistGradientBoostingClassifier": {"type": "class", "seed_param": "random_state"},
    "HistGradientBoostingRegressor": {"type": "class", "seed_param": "random_state"},
    "AdaBoostClassifier": {"type": "class", "seed_param": "random_state"},
    "AdaBoostRegressor": {"type": "class", "seed_param": "random_state"},
    "BaggingClassifier": {"type": "class", "seed_param": "random_state"},
    "BaggingRegressor": {"type": "class", "seed_param": "random_state"},

    "KMeans": {"type": "class", "seed_param": "random_state"},
    "MiniBatchKMeans": {"type": "class", "seed_param": "random_state"},
    "SpectralClustering": {"type": "class", "seed_param": "random_state"},
    "AffinityPropagation": {"type": "class", "seed_param": "random_state"},

    "PCA": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"svd_solver": ["randomized", "arpack"]}
    },
    "NMF": {"type": "class", "seed_param": "random_state"},
    "FastICA": {"type": "class", "seed_param": "random_state"},
    "LatentDirichletAllocation": {"type": "class", "seed_param": "random_state"},

    "TSNE": {"type": "class", "seed_param": "random_state"},
    "LocallyLinearEmbedding": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"method": ["modified", "hessian"]}
    },

    "SGDClassifier": {"type": "class", "seed_param": "random_state"},
    "SGDRegressor": {"type": "class", "seed_param": "random_state"},
    "SGDOneClassSVM": {"type": "class", "seed_param": "random_state"},
    "PassiveAggressiveClassifier": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"shuffle": True}
    },
    "PassiveAggressiveRegressor": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"shuffle": True}
    },
    "Perceptron": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"shuffle": True}
    },
    "LogisticRegression": {
        "type": "class",
        "seed_param": "random_state",
        "conditions": {"solver": ["sag", "saga", "liblinear"]}
    },

    "DecisionTreeClassifier": {"type": "class", "seed_param": "random_state"},
    "DecisionTreeRegressor": {"type": "class", "seed_param": "random_state"},
}


class RandomStateChecker(BaseChecker):
    """Warn when certain classes/functions are used without a random state parameter, disabling reproducibility."""

    name = "random_state_checker"
    msgs = {
        "W9001": (
            "%s instantiated without %s parameter for reproducibility",
            "missing-random-state",
            "For reproducible results, %s should be called with its '%s' parameter set, "
            "sometimes this is depended if other conditions (e.g., 'shuffle=True') are met."
        ),
    }

    def _get_keyword_value(self, keywords, arg_name):
        """Helper to get the actual value of a keyword argument in the function/class call."""
        if not keywords:
            return None
        for keyword in keywords:
            if keyword.arg == arg_name:
                if isinstance(keyword.value, astroid.Const):
                    return keyword.value.value
        return None

    def _send_message(self, node, class_name, expected_seed_param):
        """Helper to send a message when a class/function is used without the required random state parameter."""
        self.add_message(
            "missing-random-state",
            node=node,
            args=(class_name, expected_seed_param),
        )

    def _compare_conditions(self, node, expected_conditions):
        """Helper to check if the conditions for a class/function are met."""
        conditions_met = False
        num_conditions_checked = 0
        num_conditions_satisfied = 0
        for cond_key, cond_value in expected_conditions.items():
            num_conditions_checked += 1
            actual_value = self._get_keyword_value(node.keywords, cond_key)
            if isinstance(cond_value, list):
                if actual_value in cond_value:
                    num_conditions_satisfied += 1
            elif actual_value == cond_value:
                num_conditions_satisfied += 1
        if num_conditions_checked > 0 and num_conditions_satisfied == num_conditions_checked:
            conditions_met = True
        return conditions_met

    def visit_call(self, node):
        """
        This method is called by Pylint whenever it finds a function call
        or class instantiation (like `MyClass()` or `my_function()`) in the code.
        """
        # Infer the function/class being called.
        inferred_nodes = node.func.infer()
        # It returns a generator, so with next() we get the first one (usually the best and only one).
        inferred_callable = next(inferred_nodes)

        # Get the class name from the inferred callable
        class_name = inferred_callable.name
        # Check if the class is one of the target classes we care about
        if class_name in TARGETS_RANDOMNESS:
            arg_map = TARGETS_RANDOMNESS[class_name]

            if arg_map["type"] == "class" and not isinstance(inferred_callable, astroid.ClassDef):
                return
            if arg_map["type"] == "function" and not isinstance(inferred_callable, astroid.FunctionDef):
                return
            expected_seed_param = arg_map["seed_param"]
            expected_conditions = arg_map.get("conditions", {})

            conditions_met = True
            # Check if all the conditions are met
            if expected_conditions:
                conditions_met = self._compare_conditions(node, expected_conditions)

            # If the conditions are met, check if the random state parameter is present
            if conditions_met:
                seed_param_found = False
                if node.keywords:
                    for keyword in node.keywords:
                        if keyword.arg == expected_seed_param:
                            seed_param_found = True
                            break
                if not seed_param_found:
                    self._send_message(node, class_name, expected_seed_param)


def register(linter):
    """Standard pylint plugin registration hook."""

    linter.register_checker(RandomStateChecker(linter))
