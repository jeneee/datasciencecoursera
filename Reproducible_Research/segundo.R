



my.path <- "C:/r/segundo_trabajo";
setwd(my.path);

data <- read.csv("C:/r/segundo_trabajo/repdata_data_StormData.csv", header=T , sep=",")
print (data)


to_read3 = c("NULL", "character", rep("NULL", 4)
             , "character", "character", rep("NULL", 14)
             , "numeric", "numeric", "numeric", "character"
             , "numeric", "character", rep("NULL", 3),
             "numeric", "numeric", rep("NULL", 3), "numeric")

system.time(storm <- read.csv("repdata-data-StormData.csv.bz2", colClasses = to_read3))

# Read data
data <- read.csv("repdata-data-StormData.csv.bz2", header = TRUE, sep=",")

head(data)

# Find the most harm by data
harm <- data$FATALITIES + data$INJURIES
maxharm <- max(harm)
maxharm
mostharm <- data$EVTYPE[which(data$INJURIES + data$FATALITIES == maxharm)]
mostharm


tornado <- data[which(data$EVTYPE == "TORNADO"),]
par(mfrow=c(1,2))

boxplot(tornado$INJURIES, main ="Injuries caused by Tornado"
        ,xlab="Storm", ylab="Number of Injuries")

boxplot(tornado$FATALITIES, main ="Fatalities caused by Tornado"
        ,xlab="Storm", ylab="Number of Injuries")

# Prop damage
prop <- data$PROPDMG[which(data$PROPDMGEXP == "B")]
prop
a <- max(prop)

# Crop damage
crop <- data$CROPDMG[which(data$CROPDMGEXP == "B")]
crop
max(crop)

econ <- data$EVTYPE[which((data$PROPDMG == a ) & (data$PROPDMGEXP == "B"))]
econ

# Plot
boxplot(data$PROPDMG[data$EVTYPE=="FLOOD"]
        , main ="Property Damage caused by Flood"
        ,xlab="FLOOD", ylab="Billions of Dollar")
# Check R publish
getOption("rpubs.upload.method")