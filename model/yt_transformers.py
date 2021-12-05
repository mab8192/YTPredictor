from transformers import AutoTokenizer
from torchvision import transforms


# copied from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


tokenizers = {'ibert', 'layoutlm', 'mbart', 'm2m_100', 'openai-gpt', 'distilbert', 'clip', 'gpt2', 'flaubert',
              'mobilebert', 'ctrl', 'bert-japanese', 'phobert', 'gpt_neo', 'lxmert', 'byt5', 'wav2vec2', 'xlm', 'fnet',
              'rembert', 'convbert', 'roberta', 'roformer', 'rag', 'bart', 'deberta-v2', 'blenderbot',
              'blenderbot-small', 'luke', 'mt5', 'pegasus', 'speech_to_text_2', 'xlnet', 'reformer', 'bigbird_pegasus',
              'transfo-xl', 'camembert', 'led', 'retribert', 'splinter', 't5', 'canine', 'fsmt', 'mbart50',
              'speech_to_text', 'big_bird', 'xlm-roberta', 'hubert', 'cpm', 'longformer', 'mpnet', 'bartpho',
              'bertweet', 'barthez', 'prophetnet', 'layoutlmv2', 'tapas', 'deberta', 'dpr', 'funnel', 'albert',
              'marian', 'squeezebert', 'electra', 'bert-generation', 'xlm-prophetnet', 'bert'}


class TitleTransform:
    def __init__(self, model='bert') -> None:
        if model not in tokenizers:
            raise ValueError(f'Model "{model}" is not a valid tokenizer')
        self.yt_tokenizer = AutoTokenizer.from_pretrained(f'{model}-base-uncased')

    def __call__(self, *args, **kwargs):
        return self.yt_tokenizer(*args, **kwargs, return_tensors='pt')
