
import time


class SpamDetector:

    MAX_MESSAGE_LENGTH = 5000

    def __init__(self, model):
        self.model = model
        self.classes = list(model.classes_)

        if "spam" not in self.classes:
            raise ValueError(
                "The model must contain a spam class."
            )

    def predict(self, message):

        # 1. Validate input
        if not isinstance(message, str):
            raise TypeError("Message must be text.")

        message = message.strip()

        if not message:
            raise ValueError("Message cannot be empty.")

        if len(message) > self.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message exceeds {self.MAX_MESSAGE_LENGTH} characters."
            )

        # 2. Run model inference once
        start_time = time.perf_counter()

        probabilities = self.model.predict_proba([message])[0]

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # 3. Select the class with highest probability
        best_index = probabilities.argmax()

        prediction = self.classes[best_index]

        # 4. Extract class probabilities
        spam_index = self.classes.index("spam")

        spam_score = float(
            probabilities[spam_index] * 100
        )

        confidence = float(
            probabilities[best_index] * 100
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "spam_score": spam_score,
            "latency_ms": elapsed_ms
        }

    def predict_many(self, messages):

        valid_messages = []

        for message in messages:
            if not isinstance(message, str):
                raise TypeError("Every message must be text.")

            message = message.strip()

            if not message:
                raise ValueError("Messages cannot be empty.")

            if len(message) > self.MAX_MESSAGE_LENGTH:
                raise ValueError("A message is too long.")

            valid_messages.append(message)

        if not valid_messages:
            return []

        start_time = time.perf_counter()

        # One batch inference instead of one call per message
        probabilities = self.model.predict_proba(
            valid_messages
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        spam_index = self.classes.index("spam")

        results = []

        for row in probabilities:

            best_index = row.argmax()

            results.append({
                "prediction": self.classes[best_index],
                "confidence": float(row[best_index] * 100),
                "spam_score": float(row[spam_index] * 100)
            })

        return {
            "results": results,
            "total_latency_ms": elapsed_ms,
            "messages_per_second": (
                len(valid_messages) / (elapsed_ms / 1000)
                if elapsed_ms > 0 else 0
            )
        }
