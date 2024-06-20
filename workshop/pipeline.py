import pandas as pd
from typing import Tuple
from datasets import load_dataset
import mlflow

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# from config import label_names

label_names = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

class Pipeline:
    def __init__(self):
        print("Initializing sentence transformers")
        self.embeddings_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("Setting correct mlflow tracking path")
        mlflow.set_tracking_uri("file:///workspaces/build-your-first-ml-pipeline-workshop/mlruns")

        self._mlflow_model = None

    def train(self, train_data = None, test_data = None, train_embeddings = None, sample_train_n=None):
        if not isinstance(train_data, pd.DataFrame) or not isinstance(test_data, pd.DataFrame):
            train_data, test_data = self.load_dataset()

        # to be able to test the pipeline faster we sample it down
        if sample_train_n:
            print(f"Sampling the training set to a smaller quantity {sample_train_n} ")
            train_data = train_data.sample(sample_train_n)

        if not isinstance(train_embeddings, pd.DataFrame):
            train_embeddings = self.create_embeddings(train_data)

        X_train, X_val, y_train, y_val = train_test_split(
            train_embeddings, train_data['label_name'], test_size=0.2, random_state=0)

        print("Training KNN")
        mlflow.autolog()  
        
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        print(classification_report(y_val, y_pred))

        self.model = knn
        self.predict("I still haven't recieved my card, when will it be ready?")

        return self.model

    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataset =  load_dataset("PolyAI/banking77", revision="main") # taking the data from the main branch
    
        train_data = pd.DataFrame(dataset['train'])
        test_data = pd.DataFrame(dataset['test'])

        train_data["label_name"] = train_data["label"].apply(lambda x: label_names[x])
        test_data["label_name"] = test_data["label"].apply(lambda x: label_names[x])

        return train_data, test_data
    
    def create_embeddings(self, train_data):
        print("Encoding embeddings")
        
        train_text_lists = train_data.text.tolist()

        train_embeddings = self.embeddings_model.encode(train_text_lists, show_progress_bar=True)

        return train_embeddings
    
    
    def predict(self, text_input: str):
        print(f"Prediction for {text_input}")
        if not self.model:
            raise Exception("You first need to train a model use pipeline.train to do so")
        
        print(self.model.predict(self.embeddings_model.encode(text_input).reshape(1, -1)))


    def predict_mlflow_model(self, text_input: str):
        if not self._mlflow_model:
            model_id = """6e0181206f054fe3902f09f1c83f09b6"""
            self._mlflow_model = mlflow.sklearn.load_model(f"file:///workspaces/build-your-first-ml-pipeline-workshop/mlruns/0/{model_id}/artifacts/model")

        return self._mlflow_model.predict(self.embeddings_model.encode(text_input).reshape(1, -1))



if __name__ == "__main__":
    import fire
    fire.Fire(Pipeline)
