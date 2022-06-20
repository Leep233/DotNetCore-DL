using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public struct Vector2D
    {
        public const string FloatFormat = "F4";

        public float x;
        public float y;

        public  static Vector2D operator -(Vector2D v1,Vector2D v2){
            return new Vector2D() { x = v1.x - v2.x, y = v1.y - v2.y };
        }

        public static Vector2D operator *(Vector2D v1, float c)
        {
            return new Vector2D() { x = v1.x *c, y = v1.y * c };
        }

        public static Vector2D operator *(float c,Vector2D v1)
        {
            return v1*c;
        }

        public override string ToString()
        {
            return $"({x.ToString(FloatFormat)},{y.ToString(FloatFormat)})";
        }
    }
    public struct Vector3D 
    {
        public const string FloatFormat = "F4";

        public float x;
        public float y;
        public float z;

        public override string ToString()
        {
            return $"({x.ToString(FloatFormat)},{y.ToString(FloatFormat)},{z.ToString(FloatFormat)})";
        }
    }
}
