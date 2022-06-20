using System;
using System.Collections.Generic;
using System.Text;

namespace Deeplearning.Core.Math
{
    public struct Vector2D
    {
        public float x;
        public float y;

        public  static Vector2D operator -(Vector2D v1,Vector2D v2){
            return new Vector2D() { x = v1.x - v2.x, y = v1.y - v2.y };
        }
        public override string ToString()
        {
            return $"({x.ToString("f2")},{y.ToString("f2")})";
        }
    }
    public struct Vector3D 
    { 
        public float x;
        public float y;
        public float z;

        public override string ToString()
        {
            return $"({x.ToString("f2")},{y.ToString("f2")},{z.ToString("f2")})";
        }
    }
}
