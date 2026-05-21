class Modeltools:  
    # Trie un dictionnaire par ses valeurs (ordre croissant).
    # lambda est une fonction anonyme : lambda item: item[1] extrait la valeur
    # de chaque paire (clé, valeur) pour l'utiliser comme critère de tri.
    @staticmethod
    def sorted_dict_by_values(dict_object):
        sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
        return sorted_dict
    

    @staticmethod
    def evaluate(model, X, y):
        y_true = list(y)
        y_pred = model.predict(X)
        n_samples = len(y_true)

        correct = 0
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == pred_label:
                correct += 1
        accuracy = correct / n_samples

        classes = set(y_true) | set(y_pred)
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1 = 0

        for current_class in classes:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            support = 0

            for true_label, predicted_label in zip(y_true, y_pred):
                if true_label == current_class:
                    support += 1
                    if predicted_label == current_class:
                        true_positives += 1
                    else:
                        false_negatives += 1
                elif predicted_label == current_class:
                    false_positives += 1

            if true_positives + false_positives == 0:
                precision = 0.0
            else:
                precision = true_positives / (true_positives + false_positives)

            if true_positives + false_negatives == 0:
                recall = 0.0
            else:
                recall = true_positives / (true_positives + false_negatives)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            weight = support / n_samples
            weighted_precision += precision * weight
            weighted_recall += recall * weight
            weighted_f1 += f1 * weight

        return {
            "accuracy": accuracy,
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        }

    @staticmethod
    def grid_search(model_class, X_train, y_train, X_val, y_val, param_grid, score_metric="f1"):
        best_score = -float('inf')
        best_params = None
        all_results = []

        for params in Modeltools._cartesian_product(param_grid): 
            candidate = model_class(**params)
            candidate.fit(X_train, y_train)
            metrics = Modeltools.evaluate(candidate, X_val, y_val)

            score = metrics[score_metric]
            all_results.append({"params": params, "metrics": metrics})

            if score > best_score:
                best_score = score
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }