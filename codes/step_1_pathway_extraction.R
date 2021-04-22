  
#####################################
### Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
### - Associated publication:
### url: 
### doi: 
### github: 
#####################################
library( gage )

data( kegg.gs )

gse6978_anl <- read.csv( "GSE6978_gsnz.csv", quote="" )

#
# the file GSE6978_gsnz.csv is transposed from the typical input, and the
# conditions are given in the rightmost column, so converterting that
#

num_conditions <- nrow( gse6978_anl )
num_genes <- ncol( gse6978_anl ) - 1   # last column is conditions (exposures)

cat( num_conditions, " conditions   ", num_genes, " genes\n" )

conditions <- gse6978_anl[,num_genes + 1]

print( conditions )

genes <- substring( colnames( gse6978_anl )[1:num_genes], 2 )

cat( "head and tail of genes vector is:\n" )
print( head( genes ) )
print( tail( genes ) )

#
# create a transposed matrix so that rows correspond to genes
# and columns correspond to conditions (exposures)
#

tm <- t( gse6978_anl[,1:num_genes] ) 

#
# relable rows and columns
#
rownames( tm ) <- genes
colnames( tm ) <- conditions 

cat( "top left corner of tm is:\n" )
print( tm[1:5, 1:6] )


gse6978.kegg.05.p <- gage( tm, 
                           gsets = kegg.gs
                         )
cat( "str of gage result gse6978.kegg.05.p:\n" )

print( str( gse6978.kegg.05.p ) )


# run through your loop, adding each vector to the list
pathwaynames <- names( kegg.gs )

lengthTable <- 1:177
for(i in 1:177){
  temp <- unlist(kegg.gs[i], use.names = FALSE)
  lengthTable[i] <- length(temp)
}

pathwayIDs <- unlist(kegg.gs, use.names = FALSE)

write.table(lengthTable, file = "lengthTable.csv", row.names = FALSE)
write.table(pathwayIDs, file = "pathwayIDs.csv", row.names = FALSE)

