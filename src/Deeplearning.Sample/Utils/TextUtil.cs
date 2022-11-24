using Deeplearning.Core.Math;


namespace Deeplearning.Sample.Utils
{
    public class TextUtil
    {

        public static string MatrixToString(Matrix matrix) 
        { 
            return matrix.ToString();
        }
        public static Matrix StringToMatrix(string matrixStr) 
        {
            string[] rowStr = matrixStr.TrimStart().TrimEnd().Split('\n');
            int row = rowStr.Length;
            char[] chars = new char[] { ',', ' ' };

            int col = rowStr[0].TrimStart().TrimEnd().Split(chars).Length;

            Matrix matrix = new Matrix(row,col);

            for (int r = 0; r < row; r++)
            {
                string[] strs = rowStr[r].TrimStart().TrimEnd().Split(chars);

                for (int c = 0; c < strs.Length; c++)
                {
                    if (float.TryParse(strs[c], out float value)) {
                        matrix[r, c] = value;
                    }                      
                }
            }
            return matrix;
        }
    }
}
