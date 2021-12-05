from YTPredictor.model.yt_transformers import TitleTransform
from transformers import AutoModel


models = {'ibert', 'layoutlm', 'mbart', 'm2m_100', 'openai-gpt', 'distilbert', 'clip', 'gpt2', 'flaubert', 'mobilebert',
          'ctrl', 'sew-d', 'gpt_neo', 'lxmert', 'visual_bert', 'unispeech', 'wav2vec2', 'xlm', 'fnet', 'rembert',
          'convbert', 'roberta', 'roformer', 'segformer', 'bart', 'deberta-v2', 'blenderbot', 'blenderbot-small',
          'luke', 'mt5', 'pegasus', 'xlnet', 'reformer', 'unispeech-sat', 'sew', 'bigbird_pegasus', 'transfo-xl',
          'camembert', 'led', 'retribert', 'splinter', 't5', 'canine', 'fsmt', 'speech_to_text', 'vit', 'big_bird',
          'xlm-roberta', 'hubert', 'longformer', 'mpnet', 'deit', 'prophetnet', 'layoutlmv2', 'tapas', 'deberta', 'dpr',
          'funnel', 'albert', 'marian', 'squeezebert', 'beit', 'gptj', 'electra', 'bert-generation', 'xlm-prophetnet',
          'bert', 'megatron-bert', 'detr'}


class TitleFeatureExtractor:
    def __init__(self, model='bert') -> None:
        if model not in models:
            raise ValueError(f'Model "model" is not in model set')
        self.yt_model = AutoModel.from_pretrained(f'{model}-base-uncased')
        self.yt_transform = TitleTransform()

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(object.__getattribute__(self, 'yt_model'), item)
        except AttributeError:
            return object.__getattribute__(self, item)

    def __call__(self, *args, **kwargs):
        return self.yt_model.forward(**self.yt_transform(*args, **kwargs))['pooler_output']


if __name__ == '__main__':
    from YTPredictor import ThumbnailDataset
    my_model = TitleFeatureExtractor()
    data = ThumbnailDataset(root='/home/corbin/Desktop/school/fall2021/deep/final_project/YTPredictor/youtube_api/')
    txt = data[0][1]['title']
    print(f'{txt=}')
    feature = my_model(txt)
    print(f'{feature.shape=}')