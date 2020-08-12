#!/usr/bin/python3
import os
import numpy as np
import cv2
import nibabel as nib

# from gorchard @ https://github.com/gorchard/event-Python/blob/master/eventvision.py
# dvs data workings withs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """
    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)], shape=(num_events))
        self.width = width
        self.height = height

    def show_em(self,frame_length = 24e3):
        """Displays the EM events (grayscale ATIS events)"""
        frame_length = frame_length
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        max_val = 1.16e5
        min_val = 1.74e3
        val_range = max_val - min_val

        thr = np.rec.array(None, dtype=[('valid', np.bool_), ('low', np.uint64), ('high', np.uint64)], shape=(self.height, self.width))
        thr.valid.fill(False)
        thr.low.fill(frame_start)
        thr.high.fill(0)

        def show_em_frame(frame_data):
            """Prepare and show a single frame of em data to be shown"""
            for datum in np.nditer(frame_data):
                ts_val = datum['ts'].item(0)
                thr_data = thr[datum['y'].item(0), datum['x'].item(0)]

                if datum['p'].item(0) == 0:
                    thr_data.valid = 1
                    thr_data.low = ts_val
                elif thr_data.valid == 1:
                    thr_data.valid = 0
                    thr_data.high = ts_val - thr_data.low

            img = 255 * (1 - (thr.high - min_val) / (val_range))
            #thr_h = cv2.adaptiveThreshold(thr_h, 255,
            #cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
            img = np.piecewise(img, [img <= 0, (img > 0) & (img < 255), img >= 255], [0, lambda x: x, 255])
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(1)

        while frame_start < t_max:
            #with timer.Timer() as em_playback_timer:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
            show_em_frame(frame_data)
            frame_start = frame_end + 1
            frame_end += frame_length + 1
            #print 'showing em frame took %s seconds' %em_playback_timer.secs

        cv2.destroyAllWindows()
        return

    def show_td(self, wait_delay=1):
        """Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        frame_length = 24e3
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        td_img = np.ones((self.height, self.width), dtype=np.uint8)
        while frame_start < t_max:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
            
            if frame_data.size > 0:
                td_img.fill(128)

                #with timer.Timer() as em_playback_timer:
                for datum in np.nditer(frame_data):
                    td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
                #print 'prepare td frame by iterating events took %s seconds'
                #%em_playback_timer.secs

                td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])
                cv2.imshow('img', td_img)
                cv2.waitKey(wait_delay)

            frame_start = frame_end + 1
            frame_end = frame_end + frame_length + 1

        cv2.destroyAllWindows()
        return

def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = Events(td_indices.size, 34, 34)
    td.data.x = all_x[td_indices]
    td.width = td.data.x.max() + 1
    td.data.y = all_y[td_indices]
    td.height = td.data.y.max() + 1
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]
    return td
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Slayer network and dataloader definitionsing
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
torch.set_default_tensor_type(torch.FloatTensor)

# Dataset definition
class caltechDataset(Dataset):
  def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
    self.path = datasetPath 
    self.samplingTime = samplingTime
    self.samples = np.load(sampleFile,allow_pickle = True)
    self.nTimeBins    = int(sampleLength / samplingTime)
    self.classes = np.sort(np.unique(self.samples[1,:]))


  def __getitem__(self, index):
    input_index  = self.samples[0,index]
    class_label  = self.samples[1,index]
    class_code = list(self.classes).index(class_label)

    inputSpikes = snn.io.read2Dspikes(
                    input_index
                    ).toSpikeTensor(torch.zeros((2,34,34,self.nTimeBins)),
                    samplingTime=self.samplingTime)
    desiredClass = torch.zeros((len(self.classes), 1, 1, 1))

    #[batch,channel/class,dimx,dimy,spikes]
    desiredClass[class_code,...] = 1
    # should be able to view spikes to visualize different sampling times
    return inputSpikes, desiredClass, class_code

  def __len__(self):
      return len(self.samples[0,:])

# Network definition
#class Network(torch.nn.Module):
#    def __init__(self, netParams):
#        super(Network, self).__init__()
#        # initialize slayer
#        self.netParams = netParams
#        slayer = snn.layer(self.netParams['neuron'], self.netParams['simulation'])
#        self.slayer = slayer
#        self.NUM_CLASSES = self.netParams['training']['num_classes']
#        # define network functions
#        self.conv1 = slayer.conv(2, 16, 5, padding=1)
#        self.conv2 = slayer.conv(16, 32, 3, padding=1)
#        self.conv3 = slayer.conv(32, 64, 3, padding=1)
#        self.pool1 = slayer.pool(2)
#        self.pool2 = slayer.pool(2)
#        self.fc1   = slayer.dense((8, 8, 64),self.NUM_CLASSES)
#
#    def forward(self, spikeInput):
#        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput ))) # 32, 32, 16
#        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1))) # 16, 16, 16
#        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2))) # 16, 16, 32
#        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3))) #  8,  8, 32
#        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4))) #  8,  8, 64
#        spikeOut    = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer5))) #  10
#
#        return spikeOut
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SimTB Dataset

class simtbDataset(Dataset):
  def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
    self.path = datasetPath 
    self.samples = np.load(sampleFile,allow_pickle = True)
    self.classes = np.unique(self.samples[1,:])

  def __getitem__(self, index):
    input_fname  = self.samples[0,index]
    class_label  = self.samples[1,index]
    #simtb dc: A = 1,B = 2
    if(class_label == 'A'):
        class_code = 0
    else:
        class_code = 1

    class_code = np.where(self.classes == class_label)[0][0]
    
    input_spikes = nib.load(input_fname)
    input_spikes = torch.Tensor(input_spikes.get_fdata())
    input_spikes = np.tile(input_spikes.unsqueeze(0),[2,1,1,1])
    desired_class = torch.zeros((2, 1, 1, 1),dtype=torch.float32)
    desired_class[class_code,...] = 1
    # should be able to view spikes to visualize different sampling times
    return input_spikes, desired_class, class_code

  def __len__(self):
      return len(self.samples[0,:])



# simtb Network definition
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        self.netParams = netParams
        slayer = snn.layer(self.netParams['neuron'], self.netParams['simulation'])
        self.slayer = slayer
        self.NUM_CLASSES = self.netParams['training']['num_classes']
        # define network functions
        self.conv1 = slayer.conv(2, 16, 5, padding=1)
        self.conv2 = slayer.conv(16, 32, 3, padding=1)
        self.conv3 = slayer.conv(32, 64, 3, padding=1)
        self.conv4 = slayer.conv(64, 64, 3, padding=1)
        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)
        self.pool3 = slayer.pool(2)
        self.fc1   = slayer.dense((6, 6, 64),self.NUM_CLASSES)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput ))) # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1))) # 16, 16, 16
        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2))) # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3))) #  8,  8, 32
        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4))) #  8,  8, 64
        spikeLayer6 = self.slayer.spike(self.pool3(self.slayer.psp(spikeLayer5))) #  8,  8, 32
        spikeLayer7 = self.slayer.spike(self.conv4(self.slayer.psp(spikeLayer6))) #  8,  8, 64
        spikeOut    = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer7))) #  10

        return spikeOut
