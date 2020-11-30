import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from constants import SAMPLE_RATE, N_MELS, N_FFT, F_MAX, F_MIN, HOP_SIZE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, normalized=False)
    
    def forward(self, audio):
        batch_size = audio.shape[0]
        
        # alignment correction to match with pianoroll
        # pretty_midi.get_piano_roll use ceil, but torchaudio.transforms.melspectrogram uses
        # round when they convert the input into frames.
        padded_audio = nn.functional.pad(audio, (N_FFT // 2, 0), 'constant')
        mel = self.melspectrogram(audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = th.log(th.clamp(mel, min=1e-9))
        return mel



class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size * 1 channel * frames * input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)  # (B x T x C)
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)  # (B x T x C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out

class Transcriber_RNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()


        self.melspectrogram = LogMelSpectrogram()

        self.frame_lstm = nn.LSTM(input_size = 229, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        self.onset_lstm = nn.LSTM(input_size = 229, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')

        out1, _ = self.frame_lstm(mel, (frame_h0,frame_c0))
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)

        out2, _ = self.onset_lstm(mel, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2)

        return frame_out, onset_out


class Transcriber_CRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(88*2, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)
        #print(mel.shape)

        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')

        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1, _ = self.frame_lstm(out1, (frame_h0,frame_c0))
        #print(out1.shape)
        frame_out = self.frame_fc(out1)
        #print(frame_out.shape)

        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        onset_out = self.onset_fc(out2)

        return frame_out, onset_out


class Transcriber_ONF(nn.Module):  # original design with killing gradient
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc1 = nn.Linear(fc_unit, 88)
        self.frame_lstm = nn.LSTM(input_size = 88+88, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.frame_fc2 = nn.Linear(88*2, 88)

        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_lstm = nn.LSTM(input_size = fc_unit, hidden_size = 88, num_layers = 2, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(88*2, 88)


    def forward(self, audio):
        mel = self.melspectrogram(audio)


        # Set initial hidden and cell states
        frame_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        frame_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')
        onset_h0 = th.zeros(2*2, mel.size(0), 88).to('cuda') # 2 layer bidirectional num_layers = 2, hidden_size = 8*2
        onset_c0 = th.zeros(2*2, mel.size(0), 88).to('cuda')

        # onset
        #print(mel.shape)
        out2 = self.onset_conv_stack(mel)  # (B x T x C)
        #print(out2.shape)
        out2, _ = self.onset_lstm(out2, (onset_h0,onset_c0))
        #print(out2.shape)
        onset_out = self.onset_fc(out2)
        #print(onset_out.shape)

        # frame
        #print(mel.shape)
        out1 = self.frame_conv_stack(mel)  # (B x T x C)
        #print(out1.shape)
        out1 = self.frame_fc1(out1)
        #print(out1.shape)

        ### copy onset_out by killing its gradient
        onset_out_copy = onset_out.clone() # copy it on new variable
        onset_out_copy = onset_out_copy.detach() # kill the gradient on copied variable
        out1 = th.cat((out1,onset_out_copy), 2) # concatenate it

        out1, _ = self.frame_lstm(out1, (onset_h0,onset_c0))
        #print(out1.shape)
        frame_out = self.frame_fc2(out1)
        #print(frame_out.shape)


        return frame_out, onset_out