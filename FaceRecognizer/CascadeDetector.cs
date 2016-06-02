using System.Collections.Generic;
using Emgu.CV;
using System.Drawing;

namespace FRLib
{
    public class CascadeDetector : Detector
    {
        public CascadeClassifier CascadeClassifier { get; set; }
        public double ScaleFactor { get; set; } = 1.1;
        public int MinNeighbours { get; set; } = 6;
        public Size MinSize { get; set; } = default(Size);
        public Size MaxSize { get; set; } = default(Size);

        public CascadeDetector(CascadeClassifier cascadeClassifier)
        {
            CascadeClassifier = cascadeClassifier;
        }

        public List<Rectangle> Detect(Mat image)
        {
            if(image==null) return new List<Rectangle>();
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayImage, grayImage);
            Rectangle[] facesArray = (CascadeClassifier.DetectMultiScale(grayImage, ScaleFactor, MinNeighbours, MinSize, MaxSize));
            grayImage.Dispose();
            if (facesArray.Length > 0) return new List<Rectangle>(facesArray);
            else return new List<Rectangle>();
        }

        public List<Mat> Extract(Mat image)
        {
            List<Mat> extractedFaces = new List<Mat>();
            List<Rectangle> faces = Detect(image);
            foreach (var roi in faces)
            {
                extractedFaces.Add(new Mat(image, roi));
            }
            return extractedFaces;
        }

        public Mat ExtractLargest(Mat image)
        {
            List<Mat> extractedFaces = Extract(image);
            Mat largest = GetLargest(extractedFaces);
            foreach(Mat m in extractedFaces)
            {
                if(ReferenceEquals(largest, m)!=true)
                {
                    m.Dispose();
                }
            }
            return largest;
        }

        public Mat GetLargest(List<Mat> images)
        {
            if (images == null) return null;
            if (images.Count != 0)
            {
                int largestI = 0;
                int largestSize = images[0].Size.Height * images[0].Size.Width;
                for (int currentI = 1; currentI < images.Count; ++currentI)
                {
                    int currentSize = images[currentI].Size.Height * images[currentI].Size.Width;
                    if (currentSize > largestSize)
                    {
                        largestI = currentI;
                        largestSize = currentSize;
                    }
                }
                return images[largestI];
            }
            else return null;
        }
    }
}
