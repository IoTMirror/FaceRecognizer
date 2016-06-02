using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Util;
using System.IO;

namespace FRLib
{
    public class LBPHUserFaceRecognizer : UserFaceRecognizer
    {
        public string DataDirectory { get; set; }
        public int Radius { get; set; } = 1;
        public int Neighbours { get; set; } = 8;
        public int GridX { get; set; } = 8;
        public int GridY { get; set; } = 8;
        public double Threshold { get; set; } = double.MaxValue;

        public LBPHUserFaceRecognizer(string dataDirectory)
        {
            DataDirectory = dataDirectory;
        }

        public int Recognize(Mat image)
        {
            if (image == null) return -1;
            Directory.CreateDirectory(DataDirectory);
            IEnumerable<string> files = Directory.EnumerateFiles(DataDirectory, "*.lbph");
            List<int> users = new List<int>();
            foreach(string file in files)
            {
                string[] nameParts = file.Replace(".lbph", "").Split(new char[] { '/','\\' });
                int userid;
                if (int.TryParse(nameParts[nameParts.Length-1], out userid)) users.Add(userid);
            }
            return Recognize(image, users);
        }

        public int Recognize(Mat image, List<int> usersToCheck)
        {
            if (image == null) return -1;
            int bestUID = -1;
            double bestDistance=0;
            Directory.CreateDirectory(DataDirectory);
            foreach (int uid in usersToCheck)
            {
                LBPHFaceRecognizer lbph = new LBPHFaceRecognizer(Radius,Neighbours,GridX,GridY,Threshold);
                try
                {
                    lbph.Load(DataDirectory + "/" + uid + ".lbph");
                }
                catch (CvException)
                {
                    continue;
                }
                FaceRecognizer.PredictionResult result = lbph.Predict(image);
                if(result.Distance!= double.MaxValue && (bestUID == -1 || result.Distance < bestDistance))
                {
                    bestUID = result.Label;
                    bestDistance = result.Distance;
                }
                lbph.Dispose();
            }
            return bestUID;
        }

        public void RemoveUserData(int userId)
        {
            Directory.CreateDirectory(DataDirectory);
            File.Delete(DataDirectory + "/" + userId + ".lbph");
        }

        public void Train(int userId, Mat image)
        {
            Directory.CreateDirectory(DataDirectory);
            LBPHFaceRecognizer lbph =  new LBPHFaceRecognizer(Radius, Neighbours, GridX, GridY, Threshold);
            try
            {
                lbph.Load(DataDirectory + "/" + userId + ".lbph");
            }
            catch(CvException)
            {
                //nothing to do - new Face Recognizer
            }
            VectorOfMat images = new VectorOfMat();
            images.Push(image);
            VectorOfInt labels = new VectorOfInt();
            labels.Push(new int[] { userId });
            lbph.Update(images,labels);
            lbph.Save(DataDirectory + "/" +userId + ".lbph");
            lbph.Dispose();
        }
    }
}
