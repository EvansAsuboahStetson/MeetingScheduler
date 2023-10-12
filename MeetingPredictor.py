import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class MeetingScheduler:
    def __init__(self):
        pass

    @staticmethod
    def prepare_data(common_available_times_binary, suitable_dates_binary_persona1, suitable_dates_binary_persona2):
        # Flatten the binary matrices
        common_available_times_flat = np.array(common_available_times_binary).flatten()
        suitable_dates_flat_persona1 = np.array(suitable_dates_binary_persona1).flatten()
        suitable_dates_flat_persona2 = np.array(suitable_dates_binary_persona2).flatten()

        # Concatenate feature vectors separately for each persona
        feature_vectors_persona1 = np.concatenate((common_available_times_flat, suitable_dates_flat_persona1), axis=0)
        feature_vectors_persona2 = np.concatenate((common_available_times_flat, suitable_dates_flat_persona2), axis=0)

        # Create labels (1 for suitable dates, 0 for others) for each persona
        labels_persona1 = np.concatenate((np.zeros_like(common_available_times_flat), suitable_dates_flat_persona1),
                                         axis=0)
        labels_persona2 = np.concatenate((np.zeros_like(common_available_times_flat), suitable_dates_flat_persona2),
                                         axis=0)


        # Split the data into training and testing datasets (e.g., 80/20 split) for each persona
        X_train_persona1, X_test_persona1, y_train_persona1, y_test_persona1 = train_test_split(
            feature_vectors_persona1, labels_persona1, test_size=0.2, random_state=42)




        X_train_persona2, X_test_persona2, y_train_persona2, y_test_persona2 = train_test_split(
            feature_vectors_persona2, labels_persona2, test_size=0.2, random_state=42)


        return (X_train_persona1, X_test_persona1, y_train_persona1, y_test_persona1,
                X_train_persona2, X_test_persona2, y_train_persona2, y_test_persona2)

    @staticmethod
    def train_model(X_train, y_train):
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        X_train = X_train.reshape(-1, 1)
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        X_test = X_test.reshape(-1, 1)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def schedule_meeting(self, common_available_times_binary, suitable_dates_binary_persona1,
                         suitable_dates_binary_persona2):
        # Prepare data for both personas
        (X_train_persona1, X_test_persona1, y_train_persona1, y_test_persona1,
         X_train_persona2, X_test_persona2, y_train_persona2, y_test_persona2) = self.prepare_data(
            common_available_times_binary, suitable_dates_binary_persona1, suitable_dates_binary_persona2)

        # Train and evaluate models for both personas
        model_persona1 = self.train_model(X_train_persona1, y_train_persona1)
        accuracy_persona1 = self.evaluate_model(model_persona1, X_test_persona1, y_test_persona1)

        model_persona2 = self.train_model(X_train_persona2, y_train_persona2)
        accuracy_persona2 = self.evaluate_model(model_persona2, X_test_persona2, y_test_persona2)

        return accuracy_persona1, accuracy_persona2

    def predict_suitable_dates(self, common_available_times_binary, suitable_dates_binary_persona1,
                               suitable_dates_binary_persona2):
        # Prepare data for prediction
        (X_train_persona1, X_test_persona1, y_train_persona1, _,
         X_train_persona2, X_test_persona2, y_train_persona2, _) = self.prepare_data(
            common_available_times_binary, suitable_dates_binary_persona1, suitable_dates_binary_persona2)


        # Train models for both personas
        model_persona1 = self.train_model(X_train_persona1, y_train_persona1)
        model_persona2 = self.train_model(X_train_persona2, y_train_persona2)

        # Predict suitable dates
        predictions_persona1 = model_persona1.predict(X_test_persona1.reshape(-1, 1))
        predictions_persona2 = model_persona2.predict(X_test_persona2.reshape(-1, 1))

        print(predictions_persona2)
        print(predictions_persona1)

        suitable_indices_persona1 = np.where(predictions_persona1 == 1)[0]
        suitable_indices_persona2 = np.where(predictions_persona2 == 1)[0]

        return suitable_indices_persona1, suitable_indices_persona2



