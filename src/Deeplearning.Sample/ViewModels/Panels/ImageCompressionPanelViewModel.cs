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

namespace Deeplearning.Sample.ViewModels.Panels
{
    public class ImageCompressionPanelViewModel:BindableBase
    {

       

        private ImageSource soureImage;

        public ImageSource SourceImage
        {
            get { return soureImage; }
            set { soureImage = value;RaisePropertyChanged("SourceImage"); }
        }

        private ImageSource compressedImage;

        public ImageSource CompressedImage
        {
            get { return compressedImage; }
            set { compressedImage = value; RaisePropertyChanged("CompressedImage"); }
        }

        private int eigenCount=100;

        public int EigenCount
        {
            get { return eigenCount; }
            set { eigenCount = value; RaisePropertyChanged("EigenCount"); }
        }


        public DelegateCommand<string> LoadImageCommand { get; set; }

        public DelegateCommand PCACompressCommand { get; set; }

        private string imagePath;

        private byte[] pixels;
        private BitmapImage sourceBitMap;

        public ImageCompressionPanelViewModel()
        {
            imagePath =Path.Combine(System.Environment.CurrentDirectory, "Resources/img01.png");

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

              DLMath.Matrix matrix = Pixels2Matrix(pixels, 400);

              PCA pca = new PCA();

              PCAEventArgs eventArgs = await Task.Factory.StartNew(() => pca.EigFit(matrix, EigenCount));

              DLMath.Matrix compressMatrix = eventArgs.D * eventArgs.X;

              byte[] ps = Matrix2Pixels(compressMatrix);

             CompressedImage = ImageUtil.Pixels2Image(sourceBitMap.PixelWidth, sourceBitMap.PixelHeight,
             sourceBitMap.DpiX, sourceBitMap.DpiY, ps);

            compressing = false;
        }

        private byte[]  Matrix2Pixels(DLMath.Matrix matrix)
        {
            int length = matrix.Row * matrix.Column;

            byte[] pixels = new byte[length];

            for (int r = 0; r < matrix.Row; r++)
            {
                for (int c = 0; c < matrix.Column; c++)
                {
                    pixels[r * matrix.Column + c] = (byte)(matrix[r, c]*255.0);
                }
            }            

            return pixels;
        }

        private DLMath.Matrix Pixels2Matrix(byte[] pixels,int r)
        {
           int length = pixels.Length;

           int count = length / r;//1600*100

            DLMath.Matrix matrix = new DLMath.Matrix(count,r);

            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < r; j++)
                {
                    matrix[i, j] = pixels[i*r+j] / 255.0;
                }
            }
            return matrix;
        }



        private void ExecuteLoadImageCommand(string path)
        {
            if (!File.Exists(path)) return;

             byte[] imgBytes =  File.ReadAllBytes(path);

            sourceBitMap = new BitmapImage();

            sourceBitMap.BeginInit();

            sourceBitMap.StreamSource = new MemoryStream(imgBytes);
            sourceBitMap.EndInit();

            SourceImage = sourceBitMap;
        
            pixels =  ImageUtil.ReadImagePixels(sourceBitMap);     
        }

    }
}
