using System.Text;

public class SmartArray
{
    private readonly int _width;
    private readonly int _length;
    private readonly float[] _data;

    public SmartArray(int length, int width)
    {
        _width = width;
        _length = length;
        _data = new float[width * length];
    }

    public SmartArray(float[] data, int width)
    {
        _width = width;
        _length = data.Length / width;
        _data = new float[_width * _length];
        data.CopyTo((Memory<float>)_data);
    }

    public SmartArray(SmartArray array)
    {
        _width = array._width;
        _length = array._length;
        _data = new float[array._data.Length];
        array._data.CopyTo((Memory<float>)_data);
    }

    public int Length => _length;
    public int Width => _width;

    public SmartArray Rand(Random? random = null)
    {
        random = random ?? new Random();
        return Apply(_ => random.NextSingle());
    }

    public SmartArray Dot(SmartArray other)
    {
        if (Width != other.Length)
        {
            throw new InvalidOperationException($"Incompatible shape: ({this}) and ({other})");
        }
        var result = new SmartArray(Length, other.Width);
        for(var row = 0; row < result.Length; row++)
            for(var column = 0; column < result.Width; column++)
            {
                float sum = 0;
                for(var i = 0; i < Width; i++)
                {
                    sum += GetValue(row, i) * other.GetValue(i, column);
                }
                result.SetValue(row, column, sum);
            }
        return result;
    }

    private SmartArray? _transposed = null;
    public SmartArray Transpose()
    {
        if (_transposed != null) return _transposed;

        var result = new SmartArray(Width, Length);
        for(var row = 0; row < result.Length; row++)
            for(var column = 0; column < result.Width; column++)
                result.SetValue(row, column, GetValue(column, row));
        
        _transposed = result;
        return result;
    }

    public static SmartArray ApplyTwo(SmartArray first, SmartArray second, Func<float, float, float> func)
    {
        if (first.Length == second.Length && first._width >= second._width && first._width % second._width == 0)
        {
            var result = new SmartArray(first.Length, first.Width);
            for(var row = 0; row < result.Length; row++)
                for(var column = 0; column < result.Width; column++)
                {
                    result.SetValue(row, column, func(first.GetValue(row, column), second.GetValue(row, column % second._width)));
                }
            return result;
        }
        throw new NotImplementedException($"Operator got unsupported shape {first} {second}");
    }

    public static SmartArray operator+(SmartArray first, SmartArray second) => ApplyTwo(first, second, (a, b) => a + b);
    public static SmartArray operator-(SmartArray first, SmartArray second) => ApplyTwo(first, second, (a, b) => a - b);
    public static SmartArray operator*(SmartArray first, SmartArray second) => ApplyTwo(first, second, (a, b) => a * b);

    public float Sum() => _data.Aggregate(0f, (a, x) => a + x);
    public SmartArray SumRows()
    {
        var result = new SmartArray(Length, 1);
        for(var row = 0; row < Length; row++)
        {
            var sum = 0f;
            for(var column = 0; column < Width; column++)
            {
                sum+=GetValue(row, column);
            }
            result.SetValue(row, 0, sum);
        }
        return result;        
    }
    public SmartArray Apply(Func<float, float> modifier) => Apply((x, _) => modifier(x));
    public SmartArray Apply(Func<float, int, float> modifier)
    {
        var copy = new SmartArray(this);
        for(var i = 0; i < copy._data.Length; i++)
        {
            copy._data[i] = modifier(_data[i], i);
        }
        return copy;
    }


    public static SmartArray operator +(SmartArray sa, float number) => sa.Apply(x => x + number);
    public static SmartArray operator -(SmartArray sa, float number) => sa.Apply(x => x - number);
    public static SmartArray operator *(SmartArray sa, float number) => sa.Apply(x => x * number);
    public static SmartArray operator /(SmartArray sa, float number) => sa.Apply(x => x / number);

    public void SetValue(int row, int column, float value) => _data[row * _width + column] = value;
    public float GetValue(int row, int column) => _data[row * _width + column];

    public override string ToString() => $"({Length}, {Width})";

    private void DumpRow(StringBuilder sb, int row)
    {
            sb.AppendFormat("{0}", GetValue(row, 0));
            int column;
            for(column = 1; column < 3 && column < Width; column++)
            {
                sb.AppendFormat(" {0}", GetValue(row, column));
            }
            if (column < Width)
            {
                sb.Append(" ...");
            }
            for(column = column > Width - 4 ? column : Width - 4; column < Width; column++)
            {
                sb.AppendFormat(" {0}", GetValue(row, column));
            }
            sb.Append("\n");
    }
    public string Dump()
    {
        if (Length == 0 || Width == 0) return "";
        var sb = new StringBuilder();
        int row;
        for(row = 0; row < 3 && row < Length; row++)
        {
            DumpRow(sb, row);
        }
        if (row < Length)
        {
            sb.AppendLine("...");
        }
        for(row = row > Length - 4 ? row : Length - 4; row < Length; row++)
        {
            DumpRow(sb, row);
        }  
        return sb.ToString();      
    }
}
