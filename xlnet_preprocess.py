class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Converts data for sequence classification data sets."""

    def get_examples(self, filename, set_type, data_dir = '/gpfs/data/razavianlab/ehr_transformer/ICD_model/train_cleaned.csv.gz'):
    	"""Gets a collection of `InputExample`s for the data set."""
    	df = pd.read_csv(data_dir + file_name)
        return self._create_examples(df, set_type)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

     def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for i in range(df.shape[0]):
            
            guid = "%s-%s" % (set_type, i)
            text_a = df['NOTE_TEXT'].iloc[i]
            label = df['ICD_DIAGNOSIS_CODE_cleaned'].iloc[i]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
def main():
	# Section: Set device for PyTorch
	if torch.cuda.is_available():
		 # might need to update when using more than 1 GPU
		device = torch.device("cuda") 
	else:
		device = torch.device("cpu")
	processor = DataProcessor()

	#Load Training data
    train_examples = processor.get_examples( , set_type = 'train')
	




	