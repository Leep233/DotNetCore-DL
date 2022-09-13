using Deeplearning.Core.Math;
using Deeplearning.Core.Math.Linear;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Deeplearning.Sample.ViewModels
{
    public class ImageComparisonWindowViewModel : BindableBase
    {
        private ImageSource sourceImage;

        public ImageSource SourceImage
        {
            get { return sourceImage; }
            set { sourceImage = value; RaisePropertyChanged("SourceImage"); }
        }
        private ImageSource image;


        public ImageSource Image
        {
            get { return image; }
            set { image = value; RaisePropertyChanged("Image"); }
        }


        private string imgPath;

        private int dec;

        private float[,] r;
        private float[,] g;
        private float[,] b;

        public ImageComparisonWindowViewModel(string image1, int dec)
        {
            this.imgPath = image1;
            this.dec = dec;

            SourceImage = LoadImage();

            ImageCompress();
        }


        public async Task ImageCompress()
        {

            var result = await Task.Factory.StartNew<BitmapImage>(() =>
            {

                SVDResult result = MatrixDecomposition.SVD(new Core.Math.Models.Matrix(this.r), Orthogonalization.Householder);

                float[,] r = Compress(result).scalars;
                 result = MatrixDecomposition.SVD(new Core.Math.Models.Matrix(this.g), Orthogonalization.Householder);

                float[,] g = Compress(result).scalars;
                 result = MatrixDecomposition.SVD(new Core.Math.Models.Matrix(this.b), Orthogonalization.Householder);

                float[,] b = Compress(result).scalars;
                float[,] a = new float[b.GetLength(0), b.GetLength(1)];

                return Pixels2Image(r, g,b,a);
            });
            Image = result;
        }

        private Core.Math.Models.Matrix Compress(SVDResult svd, float rate = 0.8f)
        {
            var oldEigens = svd.S;
            int r = (int)(oldEigens.Row * 0.8f);
            int c = (int)(oldEigens.Column * 0.8f);

            var newEigens = oldEigens.Clip(0, 0, r, c);

            var newU = svd.U.Clip(0, 0, svd.U.Row, r);

            var newV = svd.V.Clip(0, 0, svd.V.Row, c);

            return newU * newEigens * newV.T;
        }


        ///// <summary>
        ///// 图片转换
        ///// </summary>
        ///// <param name="bitmap">bitmap格式图片</param>
        ///// <returns></returns>
        //private static BitmapImage BitmapToBitmapImage(System.Drawing.Bitmap bitmap)
        //{
        //    // 直接设置DPI
        //    bitmap.SetResolution(28, 28);
        //    BitmapImage bitmapImage = new BitmapImage();
        //    using (System.IO.MemoryStream ms = new System.IO.MemoryStream())
        //    {
        //        bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);

        //        bitmapImage.BeginInit();
        //        bitmapImage.StreamSource = ms;
        //        bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
        //        bitmapImage.EndInit();
        //        bitmapImage.Freeze();
        //    }
        //    return bitmapImage;
        //}

        private BitmapImage LoadImage()
        {
            BitmapImage bitmapImage = new BitmapImage();
            if (File.Exists(imgPath))
            {

                byte[] imgBytes = File.ReadAllBytes(imgPath);


                bitmapImage.BeginInit();
                bitmapImage.StreamSource = new System.IO.MemoryStream(imgBytes);

                bitmapImage.EndInit();

                var data = ReadImagePixels(bitmapImage);
                r = data.r;
                g = data.g;
                b = data.b;
            }
            return bitmapImage;
        }


        public BitmapImage Pixels2Image(params float[][,] colors)
        {

            int pixelWidth = r.GetLength(0);
            int pixelHeight = r.GetLength(1);
            int stride = pixelWidth * 4;
            int byteLength = pixelWidth * pixelHeight * 4;

            byte[] pixels = new byte[byteLength];

            for (int y = 0; y < pixelHeight; y++)
            {
                for (int x = 0; x < pixelWidth; x++)
                {
                    int index = y * stride + 4 * x;

                    for (int i = 0; i < colors.GetLength(0); i++)
                    {
                        pixels[index + i] = (byte)(colors[i][y, x] * 255.0);
                    }

                    //pixels[index] = (byte)(r[y, x] * 255.0);
                    //pixels[index + 1] = (byte)(g[y, x] * 255.0);
                    //pixels[index + 2] = (byte)(b[y, x] * 255.0);
                    //pixels[index + 3] = 255;
                }
            }

            BitmapImage image = new BitmapImage();
            image.BeginInit();

            image.StreamSource = new System.IO.MemoryStream(pixels);

            image.EndInit();

            return image;
        }


        private (float[,] r, float[,] g, float[,] b) ReadImagePixels(BitmapImage img)
        {

            int pixelWidth = img.PixelWidth;

            int pixelHeight = img.PixelHeight;

            int stride = img.PixelWidth * 4;
            int size = img.PixelHeight * stride;
            byte[] pixels = new byte[size];

            img.CopyPixels(pixels, stride, 0);

            float[,] r = new float[pixelWidth, pixelHeight];
            float[,] g = new float[pixelWidth, pixelHeight];
            float[,] b = new float[pixelWidth, pixelHeight];
            // float[,] a = new float[pixelWidth, pixelHeight];

            for (int y = 0; y < img.PixelHeight; y++)
            {
                for (int x = 0; x < img.PixelWidth; x++)
                {
                    int index = y * stride + 4 * x;
                    r[y, x] = pixels[index] / 255.0f;
                    g[y, x] = pixels[index + 1] / 255.0f;
                    b[y, x] = pixels[index + 2] / 255.0f;
                }
            }

            return (r, g, b);

        }

    }
}
