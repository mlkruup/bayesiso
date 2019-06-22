library("OpenML")

setOMLConfig(apikey = "yourKey")

datasets = listOMLDataSets(limit=NULL)


temp <- datasets[datasets$number.of.classes == 2 & datasets$number.of.features >= 2 &
           datasets$number.of.features <= 300 & datasets$number.of.instances >= 20000 &
             datasets$number.of.instances <= 1000000 & datasets$number.of.missing.values == 0 &
             datasets$number.of.symbolic.features == 1,c("name", "data.id", "number.of.classes",
                                                         "number.of.instances",
                                                         "number.of.features",
                                                         "number.of.numeric.features",
                                                         "majority.class.size", "status")]

temp$majority.class.percentage = temp$majority.class.size / temp$number.of.instances

temp$name <- paste(temp$name,c(1:length(temp$name)), sep="_")
dim(temp)
head(temp)

for (did in temp$data.id) {
  name <- temp[temp$data.id == did,]$name
  print(name)
  if (!file.exists(paste(name, ".csv", sep=""))) {
    result = tryCatch({
      data = getOMLDataSet(data.id = did)
      write.table(data$data, paste(name, ".csv", sep=""))
    }, warning = function(w) {
      print(w)
    }, error = function(e) {
      print(e)
    }, finally = {
      print("Nothing")
    })
  }
}
