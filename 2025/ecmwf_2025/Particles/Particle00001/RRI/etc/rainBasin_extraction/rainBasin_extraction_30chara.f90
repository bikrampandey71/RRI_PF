!  rainBasin_extraction.f90 
module m_common
  implicit none
  integer :: nx, ny, nodata
  integer, allocatable :: dir(:,:)
  integer, allocatable :: dmn(:,:)
  real(8) :: xllcorner, yllcorner 
  character*256 :: cellsize_c 
end module

program rainBasin_extraction
  use m_common
  implicit none
  character*512 infile
  parameter( infile = "rainBasin_extraction.txt" )
  
  character*100  :: ctemp
  character*256 :: line
  character*512 :: rri_input_file, rainfile, dirfile
  character*512 :: ofile_folder, home_folder, rri_input_folder
  character*512 :: ofile
  !character*9 , allocatable, save :: name_loc(:)
  character(30), allocatable, save :: name_loc(:)
  integer :: itemp, ios, status, getcwd, chdir
  integer :: k_loc, n_loc, n_accR, k_accR, k, dk
  integer :: i, j, nx_rain, ny_rain
  integer :: t, t_tmp, mint, maxt, maxt_tmp, bakt           ! [step]
  integer :: dt, min_time, max_time, sec_accR               ! [sec]
  integer, allocatable, save :: i_loc(:), j_loc(:)          !         (k_loc)
  integer, allocatable, save :: ijout_loc(:), num_dmn(:)    !         (k_loc)
  integer, allocatable, save :: hr_accR(:)                  ! [hr]    (k_accR)
  integer, allocatable, save :: rain_i(:), rain_j(:)        !         (ny,nx)
  integer, allocatable, save :: rain_i2(:), rain_j2(:)      !         (ny,nx)  interporation
  integer, allocatable, save :: time(:), time_tmp(:)        ! [sec]   (t[step])
  real(8) :: rtemp, mid_rain                                !                  interporation
  real(8) :: xllcorner_rain, yllcorner_rain
  real(8) :: cellsize_rain_x, cellsize_rain_y
  real(8) :: cellsize, cellsize_x, cellsize_y
  real(8), allocatable, save :: qp_org(:,:)                     ! (i, j)  rainCell
  real(8), allocatable, save :: qp_acc(:,:), qp_acctmp(:,:)     ! (i, j)  topoCell 時間累加
                            ! (=qp_dist) disribution：分布
  real(8), allocatable, save :: qp(:,:,:), qp_tmp(:,:,:)        ! (i, j, t) topoCell 
  real(8), allocatable, save :: qp_series(:,:)                  ! (t, k_loc) 地点上流
  real(8), allocatable, save :: qp_series_cum(:,:)              ! (t, k_loc) cumulation：累計


!***********************************************
! STEP 1 : Open Condition File
!***********************************************
write(*,*) "STEP 1: Condition File Reading"
open(1, file = infile, status = 'old')
  read(1,'(a)') rainfile                        ! L1 (RRI_Input:L3)
  write(*,'(9x,"rainfile: ",a)') trim(rainfile) 
  call InFileCheck(rainfile)

  read(1,'(a)') dirfile                         ! L2 (RRI_Input:L6)
  write(*,'(9x,"dirfile : ",a)') trim(dirfile)
  if (trim(dirfile) .ne. "-9999")  call InFileCheck(dirfile)
  
  read(1,*) xllcorner_rain                      ! L3 (RRI_Input:L14)
  read(1,*) yllcorner_rain                      ! L4 (RRI_Input:L15)
  read(1,*) cellsize_rain_x, cellsize_rain_y    ! L5 (RRI_Input:L16)
  
  read(1,'(a)') ctemp                           ! L6
  read(ctemp, *, iostat = ios) min_time, max_time ,dt, mid_rain
  if (ios .ne. 0) then
    read(ctemp, *, iostat = ios) min_time, max_time ,dt
    mid_rain = 0.0
    if (ios .ne. 0) then
    !pause " ! Error L6. Press Enter to stop."
    stop
  endif
    read(ctemp, *, iostat = ios) min_time, max_time ,dt
  endif

  read(1,'(a)') ofile_folder                    ! L7
  !フォルダ確認
  status = getcwd( home_folder )  ! ホームの絶対パス取得
  status = chdir ( trim(ofile_folder) ) ! 出力パスに移動
  if ( status .ne. 0 ) then
    !pause " ! Error L7. Press Enter to stop."
    stop
  endif
  status = chdir ( trim(home_folder) )!ホームパスに戻る
 
  read(1, *   ) n_accR          ! L8 時間雨量の数だけ
  k_loc = 0                     ! L9 地点数カウント↓
  do
    read(1,'(a)', iostat = ios) line
    read(line, *, iostat = ios) ctemp, itemp, itemp, itemp
    if (ios .ne. 0) then
      read(line, *, iostat = ios) ctemp, itemp, itemp
      if (ios .ne. 0)  exit
    endif
    k_loc = k_loc + 1
  enddo
  n_loc = k_loc                 ! L9 地点数カウント↑
  write(*,'(9x,"location: ",i0)')  n_loc
  allocate( name_loc(n_loc), i_loc(n_loc), j_loc(n_loc), ijout_loc(n_loc), num_dmn(n_loc) ) 
  allocate( hr_accR(n_accR) ) 

rewind(1)

  read(1,'(a)')                 ! L1 空読み(rainfile)
  read(1,'(a)')                 ! L2 空読み(dirfile)
  read(1,'(a)')                 ! L3 空読み(xllcorner_rain)
  read(1,'(a)')                 ! L4 空読み(yllcorner_rain)
  read(1,'(a)')                 ! L5 空読み(cellsize_rain_x,y)
  read(1,'(a)')                 ! L6 空読み(min_time, max_time ,dt, mid_rain)
  read(1,'(a)')                 ! L7 空読み(ofile_folder)
  
  read(1, *   ) n_accR, (hr_accR(k_accR), k_accR =1, n_accR)  ! L8

  do k_loc = 1 , n_loc          ! L9
    read(1,'(a)') ctemp
    read(ctemp, *, iostat = ios) name_loc(k_loc), i_loc(k_loc), j_loc(k_loc), ijout_loc(k_loc)
    if (ios .ne. 0) then
      read(ctemp, *, iostat = ios) name_loc(k_loc), i_loc(k_loc), j_loc(k_loc)
      write(*,*)name_loc(k_loc), len(name_loc(k_loc))
      write(*,*)i_loc(k_loc)
      write(*,*)j_loc(k_loc)
      ijout_loc(k_loc) = 0
    endif
  enddo
close(1)

! Check
  if (trim(dirfile).eq."-9999") then
    ctemp = "AllRain"
    do k_loc = 1, n_loc
      if ((i_loc(k_loc) .eq. -9999).and.(j_loc(k_loc) .eq. -9999)) then
        ctemp = name_loc(k_loc)
        itemp = ijout_loc(k_loc)
      else   ! if ((i_loc(k_loc) .ne. -9999).or.(j_loc(k_loc) .ne. -9999)) then
        if (ijout_loc(k_loc).eq.1) then
          write(*,'(" ! Ignore the location( i= ", i0, " , j= ", i0, " ) by dirfile:-9999.")') i_loc(k_loc), j_loc(k_loc)
          !pause " ! Press Enter to cotinue."
          ijout_loc(k_loc) = 0
        endif
      endif
    enddo
    n_loc = 1
    deallocate( name_loc, i_loc, j_loc, ijout_loc, num_dmn ) 
    allocate( name_loc(n_loc), i_loc(n_loc), j_loc(n_loc), ijout_loc(n_loc), num_dmn(n_loc) ) 
    name_loc(1) = trim(ctemp)
    i_loc(1) = -9999
    j_loc(1) = -9999
    ijout_loc(1) = itemp
  endif

if (n_accR .ne. 0)  then
  ios = 0
  do k_loc = 1, n_loc
    if (ijout_loc(k_loc).eq.1) ios = 1
  enddo
  if (ios .ne. 1) then
    write(*,*) "! No output point. Set '1' somewhere."
    !pause " ! Press Enter to stop."
    stop
  endif
endif

 
!***********************************************
! STEP 2 : Read Dir File
!***********************************************
write(*,*) ""
if (trim(dirfile) .eq. "-9999") then
  write(*,*) "STEP 2 : Skip dir file"
  nodata = -9999
  xllcorner = xllcorner_rain
  yllcorner = yllcorner_rain
  cellsize_x = cellsize_rain_x
  cellsize_y = cellsize_rain_y
  cellsize = ( cellsize_x + cellsize_y) / 2.0
 ! write(cellsize_c,*) cellsize  , " (" , cellsize_x , " , " , cellsize_y , " )"
  write(ctemp,*) cellsize
  cellsize_c = trim(adjustl(ctemp)) // " ( x= "
  write(ctemp,*) cellsize_x
  cellsize_c = trim(cellsize_c) // trim(adjustl(ctemp)) // " , y= "
  write(ctemp,*) cellsize_y
  cellsize_c = trim(cellsize_c) // trim(adjustl(ctemp)) // " )"
else
  write(*,*) "STEP 2 : Read dir file"
 ! dir file
  open(1, file = dirfile, status = "old" )
    read(1,*) ctemp, nx
    read(1,*) ctemp, ny
    read(1,*) ctemp, xllcorner
    read(1,*) ctemp, yllcorner
    read(1,*) ctemp, cellsize
    read(1,*) ctemp, nodata
    allocate( dir(ny, nx) )
    do i = 1, ny
      read(1, *) (dir(i, j), j = 1, nx)
    enddo
  close(1)
  cellsize_x = cellsize
  cellsize_y = cellsize
  write(cellsize_c,*) cellsize
endif


!***********************************************
! STEP 3 : Reading Rainfall Data
!***********************************************
write(*,*) ""
write(*,*) "STEP 3 : Rainfall Data"
write(*,*) "STEP 3-1 : Read Rainfall Data"
open(1, file = rainfile, status = 'old')
  allocate (rain_i(ny), rain_j(nx))
  if (mid_rain .ne. 0) allocate (rain_i2(ny), rain_j2(nx))  ! interpolation
  rain_i(:) = 0
  rain_j(:) = 0
  t = 0
  do
    read(1, *, iostat = ios) itemp, nx_rain, ny_rain
    if (trim(dirfile) .eq. "-9999") then
      nx = nx_rain
      ny = ny_rain
    endif
    do i = 1, ny_rain
      read(1, *, iostat = ios) (rtemp, j = 1, nx_rain)
    enddo
    if( ios.lt.0 ) exit
    t = t + 1
  enddo
  maxt_tmp = t - 1       ! because number from 0
  !write(*,*) "maxt(from0): ", maxt_tmp

  allocate( time_tmp(0:maxt_tmp), qp_org(ny_rain, nx_rain), qp_tmp(ny, nx, 0:maxt_tmp) )
  qp = 0.d0
  qp_org = 0.d0
  !qp_cum(:,:) = 0.d0
  
  do j = 1, nx
    rain_j(j) = int( (xllcorner + (dble(j) - 0.5d0) * cellsize_x - xllcorner_rain) / cellsize_rain_x ) + 1
    if (mid_rain .ne. 0) then         ! interpolation
      rain_j2(j) = rain_j(j)
      rtemp = (xllcorner + (dble(j) - 0.5d0) * cellsize_x - xllcorner_rain) / cellsize_rain_x
      if ( ( rtemp - int(rtemp) ) .le. mid_rain) then
        rain_j2(j) = rain_j(j) - 1
        if ( rain_j2(j) .lt. 1 )  rain_j2(j) = 1
      endif
      if ( ( 1.0 - (rtemp - int(rtemp)) ) .le. mid_rain) then
        rain_j2(j) = rain_j(j) + 1
        if ( rain_j2(j) .gt. ny_rain )  rain_j2(j) = ny_rain
      endif
    endif                             ! interpolation
  enddo
  do i = 1, ny
    rain_i(i) = ny_rain - int( (yllcorner + (dble(ny) - dble(i) + 0.5d0) * cellsize_y - yllcorner_rain) / cellsize_rain_y )
    if (mid_rain .ne. 0) then         ! interpolation
      rain_i2(i) = rain_i(i)
      rtemp = ( yllcorner + (dble(ny) - dble(i) + 0.5d0) * cellsize_y - yllcorner_rain) / cellsize_rain_y
      if ( ( rtemp - int(rtemp) ) .le. mid_rain ) then
        rain_i2(i) = rain_i(i) + 1
        if ( rain_i2(i) .gt. nx_rain )  rain_i2(i) = nx_rain
      endif
      if ( ( 1.0 - (rtemp - int(rtemp)) ) .lt. mid_rain ) then
        rain_i2(i) = rain_i(i) - 1
        if ( rain_i2(i) .lt. 1 )  rain_i2(i) = 1
      endif
    endif                             ! interpolation
  enddo

rewind(1)

  k = 0
  dk = 20
  do t = 0, maxt_tmp
    !write(*,*) t, "/", maxt_tmp
   ! org_Rain
    read(1, *) time_tmp(t), nx_rain, ny_rain
    do i = 1, ny_rain
      read(1, *) (qp_org(i, j), j = 1, nx_rain)
    enddo    
   ! Rain
    do j = 1, nx
      if(rain_j(j) .lt. 1 .or. rain_j(j) .gt. nx_rain ) cycle
      do i = 1, ny
        if(rain_i(i) .lt. 1 .or. rain_i(i) .gt. ny_rain ) cycle
        if (mid_rain .eq. 0) then
          qp_tmp(i, j, t) = qp_org( rain_i(i), rain_j(j) )
        else
          qp_tmp(i, j, t) = ( qp_org( rain_i(i), rain_j(j) ) + qp_org( rain_i(i), rain_j2(j) ) &
                            + qp_org( rain_i2(i), rain_j(j) ) + qp_org( rain_i2(i), rain_j2(j)) ) / 4.0
        endif
      enddo   ! i
    enddo     ! j
    if ( 100*t/maxt_tmp .ge. k ) then
      write(*, '(18x,i3,1x,(a3))') k, "%"
      k = k + dk
    endif 
  enddo       ! t
close(1)
deallocate( qp_org )

write(*,*) "STEP 3-2 : Replace the reading rainfall with dt pitch"
mint = int( min_time / dt ) 
if (max_time .eq. -9999) max_time = time_tmp(maxt_tmp)
maxt= int( max_time / dt ) 
allocate( time(mint:maxt), qp(ny, nx, mint:maxt) )
k = 0
dk = 20
do t = mint, maxt
  !write(*,*) (t - mint), "/", (maxt - mint)
  time(t) = t * dt
  do t_tmp = 0, maxt_tmp  
    if ( time(t) .le. time_tmp(t_tmp) ) then
      qp(:,:,t) = qp_tmp(:,:,t_tmp)
      exit
    endif
  enddo
  if ( 100*(t-mint)/(maxt-mint) .ge. k ) then
    write(*, '(18x,i3,1x,(a3))') k, "%"
    k = k + dk
  endif 
enddo

write(*,*) "    original rain t =", maxt_tmp,"time =", time_tmp(maxt_tmp)
write(*,*) "    recreate rain t =", maxt,"time =", time(maxt)

deallocate( time_tmp, qp_tmp )


!***********************************************
! STEP 4 : Calculation of upstream Rainfall at each location
!***********************************************
write(*,*) ""
write(*,*) "STEP 4 : Calculation of upstream Rainfall at each location"
allocate( dmn(ny, nx), qp_acc(ny, nx), qp_acctmp(ny, nx) )
allocate( qp_series(mint:maxt, 1:n_loc), qp_series_cum(mint:maxt, 1:n_loc) )

do k_loc = 1, n_loc         !k_loc
  write(*,*) ""
  write(*,*) "    Calc location", k_loc, "/", n_loc

! STEP 4-1 : Watershed extraction from dir
  write(*,*) "              Watershed extraction from dir"
  dmn = 0
  if (trim(dirfile).eq."-9999") then
    dmn = 1
  else
    if ((i_loc(k_loc).eq.-9999).and.(i_loc(k_loc).eq.-9999)) then
      where( dir .ne. nodata ) dmn = 1
    else
      i = i_loc(k_loc)
      j = j_loc(k_loc)
      if ( dir(i,j) .ne. nodata ) then
        dmn(i,j) = 1
        call find_upstream(i, j)
      else
        write(*,*) " Error! This point is nodata."
        write(*,*) "        i=", i, ",", "j=", j
        !pause "             Press Enter to stop"
        stop
      end if
    endif
  endif
  num_dmn(k_loc) = sum(dmn)

! STEP 4-2 : Calc Upstream Basin Average Rainfall
  write(*,*) "        Average Rainfall"
  qp_series(:,k_loc) = 0.d0
  qp_series_cum(:,k_loc) = 0.d0
  do t = mint, maxt
    do j = 1, nx
      do i = 1, ny
        if (dmn(i,j) .eq. 0 ) cycle
        qp_series(t,k_loc) = qp_series(t,k_loc) + qp(i, j, t) ! (mm/h) UpBasinTotal
      enddo
    enddo
    qp_series(t,k_loc)  = qp_series(t,k_loc) / num_dmn(k_loc)  ! (mm/h) UpBasinAverage
  enddo
  do t = 1, maxt
    qp_series_cum(t,k_loc)=qp_series_cum(t-1,k_loc)+qp_series(t,k_loc)/3600.d0*(time(t)- time(t-1)) ! (mm)
  enddo
        
! STEP 4-3 : Calc Upstream Basin Accumulated Rainfall
  if (n_accR .ne. 0) then
   if (ijout_loc(k_loc) .eq. 1 ) then
    write(*,*) "        Accumulated Rainfall"
    do k_accR = 1, n_accR             !k_accR
      if ( hr_accR(k_accR) .eq. -9999 ) then
        write(*,'("            Calc Total Rainfall")') 
        sec_accR = max_time - min_time
      else
        write(*,'("            Calc ", i0, " hr")') hr_accR(k_accR) 
        sec_accR = hr_accR(k_accR) * 3600
      endif
      bakt = sec_accR / dt - 1
      qp_acc = 0
      do t = mint, maxt               ! t
        if ( time(t) .lt. time(mint) + sec_accR ) cycle
        qp_acctmp = 0
            do t_tmp = t - bakt , t   ! t_acc
        do j = 1, nx                  ! j
          do i = 1, ny                ! i
            if (dmn(i,j) .eq. 0 ) cycle
              qp_acctmp(i, j) = qp_acctmp(i, j) + qp(i, j, t_tmp) * dt / 3600.d0
            if(qp_acc(i,j).lt. qp_acctmp(i,j)) qp_acc(i,j)=qp_acctmp(i,j)
          enddo                       ! i
        enddo                         ! j
            enddo                     ! t_acc
      enddo                           ! t
      where( dmn .eq.0 ) qp_acc = nodata 
     ! Output
      write(*,'("                Output")')
      if ( (bakt+1).gt.(maxt-mint) ) then
         write(*,'("              ! ", i0, " hr is more than duration ")') hr_accR(k_accR) 
      endif
      if ( hr_accR(k_accR) .eq. -9999 ) then
        ofile = trim(ofile_folder) // "/" // trim(name_loc(k_loc)) // "_totalRain.asc"
      else
        write(ctemp,'(i0)') hr_accR(k_accR)
        ofile = trim(ofile_folder) // "/" // trim(name_loc(k_loc)) // "_maxRain_" // trim(adjustl(ctemp)) // "hr.asc"
      endif
      call write_gis_real(ofile, qp_acc)
    enddo                             ! k_accR
   endif                              ! ijout_loc   STEP 4-3
  endif                               ! n_accR      STEP 4-3
enddo ! k_loc = 1, n_loc

!***********************************************
! STEP 5 : Output Upstream Basin Average Rainfall
!***********************************************
  write(*,*) ""
  write(*,*) "STEP 5 : Output Upstream Basin Average Rainfall"
  ofile = trim(ofile_folder) // "/" // "BasinAveRain.txt"
  open(2, file = ofile)
      write(2,'(a10, <n_loc>(a30))') "location", (trim(name_loc(k_loc)) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "i", (i_loc(k_loc) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "j", (j_loc(k_loc) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "Num_Cell", (num_dmn(k_loc) , k_loc =1, n_loc )
    do t = mint, maxt
      write(2,'(i10, <n_loc>f30.3)') time(t), (qp_series(t,k_loc) , k_loc =1, n_loc )
    enddo
  close(2)

 ! Cumulative Rainfll 
  ofile = trim(ofile_folder) // "/" // "BasinAveRain_Cum.txt"
  open(2, file = ofile)
      write(2,'(a10, <n_loc>(a30))') "location", (trim(name_loc(k_loc)) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "i", (i_loc(k_loc) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "j", (j_loc(k_loc) , k_loc =1, n_loc )
      write(2,'(a10, <n_loc>i30)') "Num_Cell", (num_dmn(k_loc) , k_loc =1, n_loc )
    do t = mint, maxt
      write(2,'(i10, <n_loc>f30.3)') time(t), (qp_series_cum(t,k_loc) , k_loc =1, n_loc )
    enddo
  close(2)

  
write(*,*) ""
write(*,*) "Complete!"
!pause "    Press Enter to end"

end program rainBasin_extraction


!■■■■■■■■■■■■■■■■■■■■■■■
! SUBROUTINE
!■■■■■■■■■■■■■■■■■■■■■■■
!***********************************************
! Check existence of InputFile
!***********************************************
subroutine InFileCheck(InF)
 implicit none
 integer(2) status, access
 character*512 InF
  status = access( trim(InF) , " ")    !" ":ファイルの有無をテスト
  if (status .ne. 0) then   !戻り値:status=0 >>> 正常 / status>0 >>> エラーコード
	write(*,*) " ! Error. No Input File."
    !pause '    Press Enter to stop'
    stop
  endif
endsubroutine InFileCheck

!***********************************************
! Find Upstream & Determine Domain
!***********************************************
recursive subroutine find_upstream(i0, j0)
	use m_common
	implicit none
	integer, intent(in) :: i0, j0
	integer :: i, j
	
  !dir = 1
  i = i0
  j = j0 + 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 16) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if
	
  !dir = 2
  i = i0 + 1
  j = j0 + 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 32) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if
	  
  !dir = 4
  i = i0 + 1
  j = j0
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 64) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if
		
  !dir = 8
  i = i0 + 1
  j = j0 - 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 128) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if

  !dir = 16
  i = i0
  j = j0 - 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 1) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if

  !dir = 32
  i = i0 - 1
  j = j0 - 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 2) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if
  
  !dir = 64
  i = i0 - 1
  j = j0
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 4) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if
		
  !dir = 128
  i = i0 - 1
  j = j0 + 1
  if(( 1 <= i .and. i <= ny) .and. (1 <= j .and. j <= nx)) then
    if(dir(i,j) == 8) then
      dmn(i,j) = 1
      call find_upstream(i, j)
    end if
  end if

  return
end subroutine

    
!***********************************************
! writing gis data (real)
!***********************************************
subroutine write_gis_real(fi, gis_data)
  use m_common
  implicit none
  real(8) gis_data(ny, nx)
  character*512 fi
  integer i, j
  open(2, file = fi)
    write(2,'("ncols", 10x,i0)') nx
    write(2,'("nrows", 10x,i0)') ny
    write(2,'("xllcorner ", 5x,f0.12)') xllcorner
    write(2,'("yllcorner ", 5x,f0.12)') yllcorner
    write(2,'("cellsize  ", 5x,a)') trim(adjustl(cellsize_c))
    write(2,'("NODATA_value   ", i0)') nodata
    do i = 1, ny
      do j = 1, nx
        if ( gis_data(i, j) .eq. nodata ) &
         & write(2, '(i0,1x)', advance='no') nodata
        if ( gis_data(i, j) .ne. nodata ) &
         & write(2, '(f0.3,1x)', advance='no') gis_data(i, j)
      enddo
      write(2, '()') 
   enddo
  close(2)
endsubroutine write_gis_real
