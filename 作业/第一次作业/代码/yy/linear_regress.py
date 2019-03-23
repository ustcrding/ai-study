import numpy as np
import pandas as pd
import xlrd
 


def _Lr_parameter(a,b):
    aa=np.linalg.pinv(a)
    w=aa*b
    return(w)

def main():
    excelFile = r'数据源.xls'
    df = pd.DataFrame(pd.read_excel(excelFile))
    b=df.loc[:,['用户满意度']]
    a=df.loc[:,['产品质量','产品价格','产品形象']]
    a=np.mat(a)
    b=np.mat(b)
    w=_Lr_parameter(a,b)
    print("满意度 = %.8f * 质量+ %.8f * 价格 + %.8f * 形象" % (w[0],w[1],w[2]))

if __name__ == '__main__':
    main()
exit
