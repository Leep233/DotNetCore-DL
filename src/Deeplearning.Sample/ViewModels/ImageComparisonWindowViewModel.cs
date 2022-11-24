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

        private double[,] r;
        private double[,] g;
        private double[,] b;

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
                SVDEventArgs result = Algebra.SVD(new Core.Math.Matrix(this.r));

                 double[,] r = Compress(result).scalars;
                 result = Algebra.SVD(new Core.Math.Matrix(this.g));

                 double[,] g = Compress(result).scalars;

                 result = Algebra.SVD(new Core.Math.Matrix(this.b));

                double[,] b = Compress(result).scalars;
                double[,] a = new double[b.GetLength(0), b.GetLength(1)];
                return null;
              //  return Pixels2Image(r, g,b,a);
            });
            Image = result;
        }

        private Core.Math.Matrix Compress(SVDEventArgs svd, float rate = 0.8f)
        {
            var oldEigens = svd.S;
            int r = (int)(oldEigens.Row * 0.8f);
            int c = (int)(oldEigens.Column * 0.8f);

            var newEigens = Core.Math.Matrix.Clip(oldEigens,0, 0, r, c);

            var newU = Core.Math.Matrix.Clip(svd.U,0, 0, svd.U.Row, r);

            var newV = Core.Math.Matrix.Clip(svd.V,0, 0, svd.V.Row, c);

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

               //var data = null; //ReadImagePixels(bitmapImage);
               //r = data.r;
               //g = data.g;
               //b = data.b;
            }
            return bitmapImage;
        }


      

    }
}
