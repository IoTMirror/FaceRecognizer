using Emgu.CV;

namespace FRLib
{
    public interface ImageProcessor
    {
        Mat process(Mat image);
    }
}
