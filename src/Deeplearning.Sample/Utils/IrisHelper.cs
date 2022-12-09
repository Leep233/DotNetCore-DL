using Deeplearning.Core.Math;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Deeplearning.Sample.Utils
{
    public enum IrisType 
    {
 
        Setosa,
        Versicolor,
        Virginica,
        Unknown,
    }

    public class IrisData
    {
        public Matrix Iris { get; set; }

        public Matrix Y { get; set; }
    }

    public class IrisHelper
    {

        public static IrisData LoadIrisData(string path, int count=-1)
        {


            string[] lines = File.ReadAllLines(path);

            int dataCount = count > 0 ? count : lines.Length - 1;

            int dimension = 5;

            Random r = new Random();

            int startIndex = r.Next(0, lines.Length - dataCount);

            int endIndex = startIndex + dataCount;

            Matrix x = new Matrix(dataCount, dimension);

            Matrix y = new Matrix(3,dataCount);

            for (int i = startIndex, j = 0; i < endIndex; i++, j++)
            {
                string content = lines[i + 1];
                string[] words = content.Split(',');
      
                x[j, 0] = float.Parse(words[0]);
                x[j, 1] = float.Parse(words[1]);
                x[j, 2] = float.Parse(words[2]);
                x[j, 3] = float.Parse(words[3]);

                IrisType tpye = IrisType.Unknown;

                switch (words[4])
                {
                    case "Iris-setosa":
                        tpye = IrisType.Setosa;
                        break;
                    case "Iris-versicolor":
                        tpye = IrisType.Versicolor;
                        break;
                    case "Iris-virginica":
                        tpye = IrisType.Virginica;
                        break;
                    default:         
                        break;
                }

                y[(int)tpye,j] =1;
            }
            return new IrisData() { Iris = x, Y = y };// (x,y);

        }

    }
}
