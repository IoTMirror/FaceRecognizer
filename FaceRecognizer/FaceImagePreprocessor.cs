using Emgu.CV;
using System.Drawing;

namespace FRLib
{
    public class FaceImagePreprocessor : ImageProcessor
    {
        public Size OutputImageSize { get; set; } = default(Size);        

        public Mat process(Mat image)
        {
            if (image==null) return null;
            if (OutputImageSize.Height == 0 || OutputImageSize.Width == 0) return null;
            Mat outputImage = new Mat();
            CvInvoke.Resize(image, outputImage, OutputImageSize);
            CvInvoke.CvtColor(outputImage, outputImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(outputImage, outputImage);
            return outputImage;
        }
    }
}
