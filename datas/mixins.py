from sklearn.metrics import classification_report, precision_recall_fscore_support

class MetricDataMixin():

    def calculate_metrics(self, output):
        raise NotImplemented
    
class ClassificationDataMixin(MetricDataMixin):

    def set_category(self, category):
        self.label2ids={k:i for i, k in enumerate(category)}
        self.id2labels={v:k for k,v in self.label2ids.items()}

        self.category = category

    def get_type_index(self, type_str):
        if type_str:
            type_str = type_str.strip()
        return self.label2ids.get(type_str, len(self.label2ids))
    
    def calculate_metrics(self, output):
        y_true, y_pred = [], []
        pred_text = []
        for gold, pred in output:
            y_true.append(self.get_type_index(gold))
            y_pred.append(self.get_type_index(pred))
            pred_text.append(pred)
        
        report = classification_report(y_true, y_pred,  
                                    labels=list(self.label2ids.values()), target_names=list(self.label2ids.keys()), 
                                    digits=3, zero_division=0)
        print(report)
        print("Examples", pred_text[:20])

        precision,recall,f1,_ = precision_recall_fscore_support(y_true, y_pred , average="micro")
        metrics = {
            "precision":precision*100,
            "recall":recall*100,
            "f1":f1*100
        }
        return metrics

class PredictionSaveMixin():
    def save_prediction(self):
        raise NotImplemented