from model.yt_transformers import TitleTransform
from transformers import AutoModel
import torch.nn as nn
import torch


model_set = {'ibert', 'layoutlm', 'mbart', 'm2m_100', 'openai-gpt', 'distilbert', 'clip', 'gpt2', 'flaubert', 'mobilebert',
          'ctrl', 'sew-d', 'gpt_neo', 'lxmert', 'visual_bert', 'unispeech', 'wav2vec2', 'xlm', 'fnet', 'rembert',
          'convbert', 'roberta', 'roformer', 'segformer', 'bart', 'deberta-v2', 'blenderbot', 'blenderbot-small',
          'luke', 'mt5', 'pegasus', 'xlnet', 'reformer', 'unispeech-sat', 'sew', 'bigbird_pegasus', 'transfo-xl',
          'camembert', 'led', 'retribert', 'splinter', 't5', 'canine', 'fsmt', 'speech_to_text', 'vit', 'big_bird',
          'xlm-roberta', 'hubert', 'longformer', 'mpnet', 'deit', 'prophetnet', 'layoutlmv2', 'tapas', 'deberta', 'dpr',
          'funnel', 'albert', 'marian', 'squeezebert', 'beit', 'gptj', 'electra', 'bert-generation', 'xlm-prophetnet',
          'bert', 'megatron-bert', 'detr'}


class TitleFeatureExtractor(nn.Module):
    def __init__(self, model_name='bert', fine_tune=False, dtype=torch.double) -> None:
        super().__init__()
        self.dtype = dtype
        assert model_name in model_set, f'Model "{model_name}" is not a valid pre-trained model'
        self.model = AutoModel.from_pretrained(f'{model_name}-base-uncased')
        self.model.eval()
        self.title_transform = TitleTransform(model_name)
        self.output_shape = tuple(self('sample text').shape[1:])
        if fine_tune:
            self.model.train()

    def forward(self, input):
        return self.model.forward(**self.title_transform(list(input), padding=True))['pooler_output'].type(self.dtype)  # using last_hidden_state produces variable output sizes


if __name__ == '__main__':
    import pathlib
    from model.dataset import ThumbnailDataset
    my_model = TitleFeatureExtractor()
    data = ThumbnailDataset(root=str(pathlib.Path(__file__).parent.resolve()) + '/../youtube_api/')
    txt = data[0][1]
    print(f'{txt=}')
    feature = my_model([txt,])
    print(f'{feature.shape=}')
