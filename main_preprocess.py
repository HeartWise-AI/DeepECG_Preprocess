import util_preprocess

def generate_balanced_dataset(directory="/media/data1/muse_ge/ecg_retrospective"):
  df = tinyxml2df("/media/data1/muse_ge/ecg_retrospective").read2flatten()
  data_set_with_leads = TinyGetWaveform(df).generate_dataset()
  train_x_,val_x_,test_x_, y_train,y_val,y_test= balanced_w_age_sex(data_set)

  return train_x_,val_x_,test_x_, y_train,y_val,y_test
