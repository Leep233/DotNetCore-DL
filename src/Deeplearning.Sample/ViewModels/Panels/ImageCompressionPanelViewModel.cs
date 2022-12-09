using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Media;
using System.IO;
using System.Windows.Media.Imaging;
using Deeplearning.Sample.Utils;
using DLMath = Deeplearning.Core.Math;
using Deeplearning.Core.Example;
using System.Threading.Tasks;
using Deeplearning.Core.Math.Linear;
using Deeplearning.Core.Math;
using System.Windows;


namespace Deeplearning.Sample.ViewModels.Panels
{
    public class ImageCompressionPanelViewModel : BindableBase
    {

        private ImageSource soureImage;

        public ImageSource SourceImage
        {
            get { return soureImage; }
            set { soureImage = value; RaisePropertyChanged("SourceImage"); }
        }

        private ImageSource compressedImage;

        public ImageSource CompressedImage
        {
            get { return compressedImage; }
            set { compressedImage = value; RaisePropertyChanged("CompressedImage"); }
        }

        private int eigenCount = 100;

        public int EigenCount
        {
            get { return eigenCount; }
            set { eigenCount = value; RaisePropertyChanged("EigenCount"); }
        }


        public DelegateCommand<string> LoadImageCommand { get; set; }

        public DelegateCommand PCACompressCommand { get; set; }

        private string imagePath;

        private byte [] pixels;

        private BitmapImage sourceBitMap;

        public ImageCompressionPanelViewModel()
        {
            imagePath = Path.Combine(System.Environment.CurrentDirectory, "Resources/img01.jpeg");

            LoadImageCommand = new DelegateCommand<string>(ExecuteLoadImageCommand);

            LoadImageCommand.Execute(imagePath);

            PCACompressCommand = new DelegateCommand(ExecutePCACompressCommand);
        }

        private bool compressing = false;

        private async void ExecutePCACompressCommand()
        {
            if (pixels is null || pixels.Length <= 0 || sourceBitMap is null || compressing) return;

            compressing = true;

            CompressedImage = null;

            int r = 400;

            DLMath.Matrix matrix = Pixels2Matrix(pixels, r);

            var result = matrix.Standardized();

            DLMath.Matrix source = result.matrix; // matrix;// 

            DLMath.Matrix D = await Task.Factory.StartNew(() => {
                //2.协方差矩阵
                //  DLMath.Matrix covMatrix = source.Cov();

                //3.对协方差矩阵求特征值特征向量
                //    EigenDecompositionEventArgs result = Algebra.Eig(covMatrix);

                SVDEventArgs result = Algebra.SVD(source);

                DLMath.Matrix eigenVectors = result.V;// result.eigenVectors;// 

                int dimension = eigenVectors.Row;      

                //4.选取有效的特征值
                DLMath.Matrix D = DLMath.Matrix.Clip(eigenVectors, 0, 0, dimension, EigenCount);

                return D;

            });

            DLMath.Matrix X = DLMath.Matrix.Dot(source, D);

            DLMath.Matrix compressMatrix = (DLMath.Matrix.Dot(D, X.T) + result.means[0]) * result.stds[0];

            byte[] ps = Matrix2Pixels(compressMatrix, r);

            CompressedImage = ImageUtil.BitmapSourceFromArray(ps,sourceBitMap.PixelWidth, sourceBitMap.PixelHeight);

            compressing = false;
        }



        private byte[] Matrix2Pixels(DLMath.Matrix matrix, int r)
        {
            int length = matrix.Row * matrix.Column;

            byte[] pixels = new byte[length];


            for (int i = 0; i < length; i++)
            {
                int x = i / r;
                int y = i % r;

                pixels[i] = (byte)matrix[y, x];
            }

            return pixels;
        }

        private DLMath.Matrix Pixels2Matrix(byte [] pixels, int r)
        {
            int length = pixels.Length;

            int count = length / r;//1600*100

            DLMath.Matrix matrix = new DLMath.Matrix(count, r);

            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < r; j++)
                {
                    matrix[i, j] = pixels[i * r + j];
                }
            }

            return matrix;
        }


        private void ExecuteLoadImageCommand(string path)
        {
            if (!File.Exists(path)) return;

            byte[] imgBytes = File.ReadAllBytes(path);

            sourceBitMap = new BitmapImage();

            sourceBitMap.BeginInit();

            sourceBitMap.StreamSource = new MemoryStream(imgBytes);

            sourceBitMap.EndInit();

            SourceImage = sourceBitMap;

            pixels = ImageUtil.BitmapSourceToArray(sourceBitMap);
        }       
    }
}
