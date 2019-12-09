import torch

from models.base_model import BaseModel
from util.util import tensor2im


################## SeasonTransfer #############################
class ABModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def prepare_data(self, data):
        img, attr_source = data
        img = torch.cat(img, 0).to(self.device)
        batch_size = img.size(0)
        attr_source = torch.cat(attr_source, 0).to(self.device)
        index_target = torch.tensor(range(-batch_size // 2, batch_size // 2)).to(self.device)
        weight_source = torch.ones([batch_size, 1]).to(self.device)
        self.current_data = [img, attr_source, index_target, weight_source]
        return self.current_data

    def translation(self, data):
        with torch.no_grad():
            self.prepare_data(data)
            img, attr_source, index_target, _ = self.current_data
            batch_size = img.size(0)
            assert batch_size == 2
            style_enc, _, _ = self.enc_style(img)
            style_target_enc = style_enc[index_target]
            attr_target = attr_source[index_target]
            content = self.enc_content(img)
            results_a2b, results_b2a = [('input_a', tensor2im(img[0].data))], [
                ('input_b', tensor2im(img[1].data))]
            fakes = self.dec(content, torch.cat([attr_target, style_target_enc], dim=1))
            results_a2b.append(('a2b_enc', tensor2im(fakes[0].data)))
            results_b2a.append(('b2a_enc', tensor2im(fakes[1].data)))
            for i in range(self.opt.n_samples):
                style_rand = self.sample_latent_code(style_enc.size())
                fakes = self.dec(content, torch.cat([attr_target, style_rand], dim=1))
                results_a2b.append(('a2b_rand_{}'.format(i + 1), tensor2im(fakes[0].data)))
                results_b2a.append(('b2a_rand_{}'.format(i + 1), tensor2im(fakes[1].data)))
            return results_a2b + results_b2a
